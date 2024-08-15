import os
import glob
import traceback
import asyncio
import json
import re
import urllib.request
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings
from typing import List, Dict, Tuple, Optional
from pdf2image import convert_from_path
import pytesseract
import tiktoken
import numpy as np
from PIL import Image
import cv2
from filelock import FileLock, Timeout
from openai import AsyncOpenAI

try:
    import nvgpu

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


API_PROVIDER = "OPENAI"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", default="your-openai-api-key")
CLAUDE_MAX_TOKENS = 4096  # Maximum allowed tokens for Claude API
TOKEN_BUFFER = 500  # Buffer to account for token estimation inaccuracies
TOKEN_CUSHION = 300  # Don't use the full max tokens to avoid hitting the limit
OPENAI_COMPLETION_MODEL = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_MAX_TOKENS = 12000  # Maximum allowed tokens for OpenAI API
USE_VERBOSE = False

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LLMOCR:

    # GPU Check
    def is_gpu_available(self):
        if not GPU_AVAILABLE:
            logging.warning("GPU support not available: nvgpu module not found")
            return {
                "gpu_found": False,
                "num_gpus": 0,
                "first_gpu_vram": 0,
                "total_vram": 0,
                "error": "nvgpu module not found",
            }
        try:
            gpu_info = nvgpu.gpu_info()
            num_gpus = len(gpu_info)
            if num_gpus == 0:
                logging.warning("No GPUs found on the system")
                return {"gpu_found": False, "num_gpus": 0, "first_gpu_vram": 0, "total_vram": 0}
            first_gpu_vram = gpu_info[0]["mem_total"]
            total_vram = sum(gpu["mem_total"] for gpu in gpu_info)
            logging.info(f"GPU(s) found: {num_gpus}, Total VRAM: {total_vram} MB")
            return {
                "gpu_found": True,
                "num_gpus": num_gpus,
                "first_gpu_vram": first_gpu_vram,
                "total_vram": total_vram,
                "gpu_info": gpu_info,
            }
        except Exception as e:
            logging.error(f"Error checking GPU availability: {e}")
            return {"gpu_found": False, "num_gpus": 0, "first_gpu_vram": 0, "total_vram": 0, "error": str(e)}

    # API Interaction Functions
    async def generate_completion(self, prompt: str, max_tokens: int = 5000) -> Optional[str]:
        if API_PROVIDER == "OPENAI":
            return await self.generate_completion_from_openai(prompt, max_tokens)
        else:
            logging.error(f"Invalid API_PROVIDER: {API_PROVIDER}")
            return None

    def get_tokenizer(self, model_name: str):
        if model_name.lower().startswith("gpt-"):
            return tiktoken.encoding_for_model(model_name)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def estimate_tokens(self, text: str, model_name: str) -> int:
        try:
            tokenizer = self.get_tokenizer(model_name)
            return len(tokenizer.encode(text))
        except Exception as e:
            logging.warning(f"Error using tokenizer for {model_name}: {e}. Falling back to approximation.")
            return self.approximate_tokens(text)

    def approximate_tokens(self, text: str) -> int:
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text.strip())
        # Split on whitespace and punctuation, keeping punctuation
        tokens = re.findall(r"\b\w+\b|\S", text)
        count = 0
        for token in tokens:
            if token.isdigit():
                count += max(1, len(token) // 2)  # Numbers often tokenize to multiple tokens
            elif re.match(r"^[A-Z]{2,}$", token):  # Acronyms
                count += len(token)
            elif re.search(r"[^\w\s]", token):  # Punctuation and special characters
                count += 1
            elif len(token) > 10:  # Long words often split into multiple tokens
                count += len(token) // 4 + 1
            else:
                count += 1
        # Add a 10% buffer for potential underestimation
        return int(count * 1.1)

    def chunk_text(self, text: str, max_chunk_tokens: int, model_name: str) -> List[str]:
        chunks = []
        tokenizer = self.get_tokenizer(model_name)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        current_chunk = []  # type: ignore
        current_chunk_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(tokenizer.encode(sentence))
            if current_chunk_tokens + sentence_tokens > max_chunk_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_chunk_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_chunk_tokens += sentence_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        adjusted_chunks = self.adjust_overlaps(chunks, tokenizer, max_chunk_tokens)
        return adjusted_chunks

    def split_long_sentence(self, sentence: str, max_tokens: int, model_name: str) -> List[str]:
        words = sentence.split()
        chunks = []
        current_chunk = []
        current_chunk_tokens = 0
        tokenizer = self.get_tokenizer(model_name)

        for word in words:
            word_tokens = len(tokenizer.encode(word))
            if current_chunk_tokens + word_tokens > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_chunk_tokens = word_tokens
            else:
                current_chunk.append(word)
                current_chunk_tokens += word_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def adjust_overlaps(self, chunks: List[str], tokenizer, max_chunk_tokens: int, overlap_size: int = 50) -> List[str]:
        adjusted_chunks = []
        for i in range(len(chunks)):
            if i == 0:
                adjusted_chunks.append(chunks[i])
            else:
                overlap_tokens = len(tokenizer.encode(" ".join(chunks[i - 1].split()[-overlap_size:])))
                current_tokens = len(tokenizer.encode(chunks[i]))
                if overlap_tokens + current_tokens > max_chunk_tokens:
                    overlap_adjusted = chunks[i].split()[:-overlap_size]
                    adjusted_chunks.append(" ".join(overlap_adjusted))
                else:
                    adjusted_chunks.append(" ".join(chunks[i - 1].split()[-overlap_size:] + chunks[i].split()))

        return adjusted_chunks

    async def generate_completion_from_openai(self, prompt: str, max_tokens: int = 5000) -> Optional[str]:
        if not OPENAI_API_KEY:
            logging.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            return None
        prompt_tokens = self.estimate_tokens(prompt, OPENAI_COMPLETION_MODEL)
        adjusted_max_tokens = min(
            max_tokens, 4096 - prompt_tokens - TOKEN_BUFFER
        )  # 4096 is typical max for GPT-3.5 and GPT-4
        if adjusted_max_tokens <= 0:
            logging.warning("Prompt is too long for OpenAI API. Chunking the input.")
            chunks = self.chunk_text(prompt, OPENAI_MAX_TOKENS - TOKEN_CUSHION, OPENAI_COMPLETION_MODEL)
            results = []
            for chunk in chunks:
                try:
                    response = await openai_client.chat.completions.create(
                        model=OPENAI_COMPLETION_MODEL,
                        messages=[{"role": "user", "content": chunk}],
                        max_tokens=adjusted_max_tokens,
                        temperature=0.7,
                    )
                    result = response.choices[0].message.content
                    results.append(result)
                    logging.info(f"Chunk processed. Output tokens: {response.usage.completion_tokens:,}")
                except Exception as e:
                    logging.error(f"An error occurred while processing a chunk: {e}")
            return " ".join(results)
        else:
            try:
                response = await openai_client.chat.completions.create(
                    model=OPENAI_COMPLETION_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=adjusted_max_tokens,
                    temperature=0.7,
                )
                output_text = response.choices[0].message.content
                logging.info(f"Total tokens: {response.usage.total_tokens:,}")
                logging.info(f"Generated output (abbreviated): {output_text[:150]}...")
                return output_text
            except Exception as e:
                logging.error(f"An error occurred while requesting from OpenAI API: {e}")
                return None

    # Image Processing Functions
    def preprocess_image(self, image):
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        kernel = np.ones((1, 1), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=1)
        return Image.fromarray(gray)

    def convert_pdf_to_images(
        self, input_pdf_file_path: str, max_pages: int = 0, skip_first_n_pages: int = 0
    ) -> List[Image.Image]:
        logging.info(f"Processing PDF file {input_pdf_file_path}")
        if max_pages == 0:
            last_page = None
            logging.info("Converting all pages to images...")
        else:
            last_page = skip_first_n_pages + max_pages
            logging.info(f"Converting pages {skip_first_n_pages + 1} to {last_page}")
        first_page = skip_first_n_pages + 1  # pdf2image uses 1-based indexing
        images = convert_from_path(input_pdf_file_path, first_page=first_page, last_page=last_page)
        logging.info(f"Converted {len(images)} pages from PDF file to images.")
        return images

    def ocr_image(self, image):
        # preprocessed_image = self.preprocess_image(image)
        return pytesseract.image_to_string(image)

    async def process_chunk(
        self,
        chunk: str,
        prev_context: str,
        chunk_index: int,
        total_chunks: int,
        reformat_as_markdown: bool,
        suppress_headers_and_page_numbers: bool,
    ) -> Tuple[str, str]:
        logging.info(f"Processing chunk {chunk_index + 1}/{total_chunks} (length: {len(chunk):,} characters)")

        # Step 1: OCR Correction
        ocr_correction_prompt = f"""Correct OCR-induced errors in the text, ensuring it flows coherently with the previous context. Follow these guidelines:
    1. Fix OCR-induced typos and errors:
    - Correct words split across line breaks
    - Fix common OCR errors (e.g., 'rn' misread as 'm', 'i' misread as 'l' or 't', '$' missed, '.' as ',', etc.)
    - Use context and common sense to correct errors
    - Only fix clear errors, don't alter the content unnecessarily
    - Do not add extra periods or any unnecessary punctuation
    2. Maintain original structure:
    - Keep all headings and subheadings intact
    3. Preserve original content:
    - Keep all important information from the original text
    - Do not add any new information not present in the original text
    - Remove unnecessary line breaks within sentences or paragraphs
    - Maintain paragraph breaks
    - Do not capitalize or change the case of any text, only grammar and spelling errors are allowed
    
    4. Maintain coherence:
    - Ensure the content connects smoothly with the previous context
    - Handle text that starts or ends mid-sentence appropriately
    IMPORTANT: Respond ONLY with the corrected text. Preserve all original formatting, including line breaks. Do not include any introduction, explanation, or metadata.
    If there is no text entered, respond with no output at all (i.e., just a blank line).
    Previous context:
    {prev_context[-500:]}
    Current chunk to process:
    {chunk}
    Corrected text:
    """

        ocr_corrected_chunk = await self.generate_completion(ocr_correction_prompt, max_tokens=len(chunk) + 500)

        processed_chunk = ocr_corrected_chunk

        # Step 2: Markdown Formatting (if requested)
        if reformat_as_markdown:
            markdown_prompt = f"""Reformat the following text as markdown, improving readability while preserving the original structure. Follow these guidelines:
    1. Preserve all original headings, converting them to appropriate markdown heading levels (# for main titles, ## for subtitles, etc.)
    - Ensure each heading is on its own line
    - Add a blank line before and after each heading
    2. Maintain the original paragraph structure. Remove all breaks within a word that should be a single word (for example, "cor- rect" should be "correct")
    3. Format lists properly (unordered or ordered) if they exist in the original text
    4. Use emphasis (*italic*) and strong emphasis (**bold**) where appropriate, based on the original formatting
    5. Preserve all original content and meaning
    6. Do not add any extra punctuation or modify the existing punctuation
    7. Remove any spuriously inserted introductory text such as "Here is the corrected text:" that may have been added by the LLM and which is obviously not part of the original text.
    8. Remove any obviously duplicated content that appears to have been accidentally included twice. Follow these strict guidelines:
    - Remove only exact or near-exact repeated paragraphs or sections within the main chunk.
    - Consider the context (before and after the main chunk) to identify duplicates that span chunk boundaries.
    - Do not remove content that is simply similar but conveys different information.
    - Preserve all unique content, even if it seems redundant.
    - Ensure the text flows smoothly after removal.
    - Do not add any new content or explanations.
    - If no obvious duplicates are found, return the main chunk unchanged.
    9. {"Identify but do not remove headers, footers, or page numbers. Instead, format them distinctly, e.g., as blockquotes." if not suppress_headers_and_page_numbers else "Carefully remove headers, footers, and page numbers while preserving all other content."}
    Text to reformat:
    {ocr_corrected_chunk}
    Reformatted markdown:
    """
            processed_chunk = await self.generate_completion(markdown_prompt, max_tokens=len(ocr_corrected_chunk) + 500)
        new_context = processed_chunk[-1000:]  # Use the last 1000 characters as context for the next chunk
        logging.info(
            f"Chunk {chunk_index + 1}/{total_chunks} processed. Output length: {len(processed_chunk):,} characters"
        )
        return processed_chunk, new_context

    async def process_chunks(
        self, chunks: List[str], reformat_as_markdown: bool, suppress_headers_and_page_numbers: bool
    ) -> List[str]:
        total_chunks = len(chunks)

        async def process_chunk_with_context(chunk: str, prev_context: str, index: int) -> Tuple[int, str, str]:
            processed_chunk, new_context = await self.process_chunk(
                chunk, prev_context, index, total_chunks, reformat_as_markdown, suppress_headers_and_page_numbers
            )
            return index, processed_chunk, new_context

        logging.info("Using API-based LLM. Processing chunks concurrently while maintaining order...")
        tasks = [process_chunk_with_context(chunk, "", i) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)
        # Sort results by index to maintain order
        sorted_results = sorted(results, key=lambda x: x[0])
        processed_chunks = [chunk for _, chunk, _ in sorted_results]
        logging.info(f"All {total_chunks} chunks processed successfully")
        return processed_chunks

    async def process_document(
        self,
        list_of_extracted_text_strings: List[str],
        reformat_as_markdown: bool = True,
        suppress_headers_and_page_numbers: bool = True,
    ) -> str:
        logging.info(f"Starting document processing. Total pages: {len(list_of_extracted_text_strings):,}")
        full_text = "\n\n".join(list_of_extracted_text_strings)
        logging.info(f"Size of full text before processing: {len(full_text):,} characters")
        chunk_size, overlap = 8000, 10
        # Improved chunking logic
        paragraphs = re.split(r"\n\s*\n", full_text)
        chunks = []
        current_chunk = []
        current_chunk_length = 0
        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            if current_chunk_length + paragraph_length <= chunk_size:
                current_chunk.append(paragraph)
                current_chunk_length += paragraph_length
            else:
                # If adding the whole paragraph exceeds the chunk size,
                # we need to split the paragraph into sentences
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                current_chunk = []
                current_chunk_length = 0
                for sentence in sentences:
                    sentence_length = len(sentence)
                    if current_chunk_length + sentence_length <= chunk_size:
                        current_chunk.append(sentence)
                        current_chunk_length += sentence_length
                    else:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                        current_chunk = [sentence]
                        current_chunk_length = sentence_length
        # Add any remaining content as the last chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk) if len(current_chunk) > 1 else current_chunk[0])
        # Add overlap between chunks
        for i in range(1, len(chunks)):
            overlap_text = chunks[i - 1].split()[-overlap:]
            chunks[i] = " ".join(overlap_text) + " " + chunks[i]
        logging.info(f"Document split into {len(chunks):,} chunks. Chunk size: {chunk_size:,}, Overlap: {overlap:,}")
        processed_chunks = await self.process_chunks(chunks, reformat_as_markdown, suppress_headers_and_page_numbers)
        final_text = "".join(processed_chunks)
        logging.info(f"Size of text after combining chunks: {len(final_text):,} characters")
        logging.info(f"Document processing complete. Final text length: {len(final_text):,} characters")
        return final_text

    async def process_image(
        self,
        extracted_string: str,
        reformat_as_markdown: bool = False,
        suppress_headers_and_page_numbers: bool = True,
    ) -> str:
        logging.info(f"Starting document processing. Total pages: {len(list_of_extracted_text_strings):,}")
        full_text = extracted_string
        logging.info(f"Size of full text before processing: {len(full_text):,} characters")
        chunk_size, overlap = 8000, 10
        # Improved chunking logic
        paragraphs = re.split(r"\n\s*\n", full_text)
        chunks = []
        current_chunk = []
        current_chunk_length = 0
        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            if current_chunk_length + paragraph_length <= chunk_size:
                current_chunk.append(paragraph)
                current_chunk_length += paragraph_length
            else:
                # If adding the whole paragraph exceeds the chunk size,
                # we need to split the paragraph into sentences
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                current_chunk = []
                current_chunk_length = 0
                for sentence in sentences:
                    sentence_length = len(sentence)
                    if current_chunk_length + sentence_length <= chunk_size:
                        current_chunk.append(sentence)
                        current_chunk_length += sentence_length
                    else:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                        current_chunk = [sentence]
                        current_chunk_length = sentence_length
        # Add any remaining content as the last chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk) if len(current_chunk) > 1 else current_chunk[0])
        # Add overlap between chunks
        for i in range(1, len(chunks)):
            overlap_text = chunks[i - 1].split()[-overlap:]
            chunks[i] = " ".join(overlap_text) + " " + chunks[i]
        logging.info(f"Document split into {len(chunks):,} chunks. Chunk size: {chunk_size:,}, Overlap: {overlap:,}")
        processed_chunks = await self.process_chunks(chunks, reformat_as_markdown, suppress_headers_and_page_numbers)
        final_text = "".join(processed_chunks)
        logging.info(f"Size of text after combining chunks: {len(final_text):,} characters")
        logging.info(f"Document processing complete. Final text length: {len(final_text):,} characters")
        return final_text

    def remove_corrected_text_header(self, text):
        return (
            text.replace("# Corrected text\n", "")
            .replace("# Corrected text:", "")
            .replace("\nCorrected text", "")
            .replace("Corrected text:", "")
        )

    async def assess_output_quality(self, original_text, processed_text):
        max_chars = 15000  # Limit to avoid exceeding token limits
        available_chars_per_text = max_chars // 2  # Split equally between original and processed

        original_sample = original_text[:available_chars_per_text]
        processed_sample = processed_text[:available_chars_per_text]

        prompt = f"""Compare the following samples of original OCR text with the processed output and assess the quality of the processing. Consider the following factors:
    1. Accuracy of error correction
    2. Improvement in readability
    3. Preservation of original content and meaning
    4. Appropriate use of markdown formatting (if applicable)
    5. Removal of hallucinations or irrelevant content
    Original text sample:
    ```
    {original_sample}
    ```
    Processed text sample:
    ```
    {processed_sample}
    ```
    Provide a quality score between 0 and 100, where 100 is perfect processing. Also provide a brief explanation of your assessment.
    Your response should be in the following format:
    SCORE: [Your score]
    EXPLANATION: [Your explanation]
    """

        response = await self.generate_completion(prompt, max_tokens=1000)

        try:
            lines = response.strip().split("\n")
            score_line = next(line for line in lines if line.startswith("SCORE:"))
            score = int(score_line.split(":")[1].strip())
            explanation = (
                "\n".join(line for line in lines if line.startswith("EXPLANATION:")).replace("EXPLANATION:", "").strip()
            )
            logging.info(f"Quality assessment: Score {score}/100")
            logging.info(f"Explanation: {explanation}")
            return score, explanation
        except Exception as e:
            logging.error(f"Error parsing quality assessment response: {e}")
            logging.error(f"Raw response: {response}")
            return None, None

    async def read_text(self, image) -> str:
        try:
            # Suppress HTTP request logs
            logging.getLogger("httpx").setLevel(logging.WARNING)
            # input_pdf_file_path = "160301289-Warren-Buffett-Katharine-Graham-Letter.pdf"
            reformat_as_markdown = False
            suppress_headers_and_page_numbers = True

            logging.info(f"Using API for completions: {API_PROVIDER}")
            logging.info(f"Using OpenAI model for embeddings: {OPENAI_EMBEDDING_MODEL}")

            # base_name = os.path.splitext(input_pdf_file_path)[0]
            # output_extension = ".md" if reformat_as_markdown else ".txt"

            # raw_ocr_output_file_path = f"{base_name}__raw_ocr_output.txt"
            # llm_corrected_output_file_path = base_name + "_llm_corrected" + output_extension

            # list_of_scanned_images = self.convert_pdf_to_images(input_pdf_file_path, max_test_pages, skip_first_n_pages)
            logging.info(f"Tesseract version: {pytesseract.get_tesseract_version()}")
            logging.info("Extracting text from image...")
            raw_ocr_output = self.ocr_image(image)
            logging.info("raw")
            logging.info(raw_ocr_output)
            # list_of_extracted_text_strings = list(executor.map(self.ocr_image, list_of_scanned_images))
            logging.info("Done extracting text from image.")
            # raw_ocr_output = "\n".join(list_of_extracted_text_strings)
            # with open(raw_ocr_output_file_path, "w") as f:
            #     f.write(raw_ocr_output)
            # logging.info(f"Raw OCR output written to: {raw_ocr_output_file_path}")

            logging.info("Processing document...")
            final_text = await self.process_document(
                [raw_ocr_output], reformat_as_markdown, suppress_headers_and_page_numbers
            )
            cleaned_text = self.remove_corrected_text_header(final_text)
            logging.info("cleaned")
            logging.info(cleaned_text)
            return cleaned_text
            # Save the LLM corrected output
            # with open(llm_corrected_output_file_path, "w") as f:
            #     f.write(cleaned_text)
            # logging.info(f"LLM Corrected text written to: {llm_corrected_output_file_path}")

            # if final_text:
            #     logging.info(f"First 500 characters of LLM corrected processed text:\n{final_text[:500]}...")
            # else:
            #     logging.warning("final_text is empty or not defined.")

            # logging.info(f"Done processing {input_pdf_file_path}.")
            # logging.info("\nSee output files:")
            # logging.info(f" Raw OCR: {raw_ocr_output_file_path}")
            # logging.info(f" LLM Corrected: {llm_corrected_output_file_path}")

            # Perform a final quality check
            # quality_score, explanation = await assess_output_quality(raw_ocr_output, final_text)
            # if quality_score is not None:
            #     logging.info(f"Final quality score: {quality_score}/100")
            #     logging.info(f"Explanation: {explanation}")
            # else:
            #     logging.warning("Unable to determine final quality score.")
        except Exception as e:
            logging.error(f"An error occurred in the main function: {e}")
            logging.error(traceback.format_exc())
        return ""
