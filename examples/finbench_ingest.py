import boto3
from typing import Optional
from pyarrow.filesystem import FileSystem
from pyarrow import fs
from sycamore.connectors.file.file_scan import JsonManifestMetadataProvider
from sycamore.functions import HuggingFaceTokenizer
from sycamore.llms import OpenAI, OpenAIModels
from sycamore.llms.prompts.default_prompts import SimpleGuidancePrompt
from sycamore.reader import DocSetReader
from sycamore.transforms.embed import SentenceTransformerEmbedder, OpenAIEmbedder
from sycamore.transforms import COALESCE_WHITESPACE
from sycamore.transforms.extract_table import CachedTextractTableExtractor, TextractTableExtractor
from sycamore.transforms.merge_elements import GreedyTextElementMerger
from sycamore.transforms.partition import SycamorePartitioner
import sycamore
from time import time
from pathlib import Path

from ray.data import ActorPoolStrategy

###########################

def get_fs():
    return fs.LocalFileSystem()

def get_s3_fs():
    session = boto3.session.Session()
    credentials = session.get_credentials()
    from pyarrow.fs import S3FileSystem

    fs = S3FileSystem(
        secret_key=credentials.secret_key,
        access_key=credentials.access_key,
        region=session.region_name,
        session_token=credentials.token,
    )
    return fs

class ManifestReader(DocSetReader):
    def binary(
        self,
        binary_format: str,
        parallelism: Optional[int] = None,
        filesystem: Optional[FileSystem] = None,
        metadata_provider: Optional[JsonManifestMetadataProvider] = None,
        file_range: Optional[list] = None,
        **resource_args
    ):
        paths = metadata_provider.get_paths()
        paths=paths if file_range == None else paths[file_range[0]:file_range[1]]
        return super().binary(
            paths=paths,
            binary_format=binary_format,
            parallelism=parallelism,
            filesystem=filesystem,
            metadata_provider=metadata_provider,
            **resource_args
        )

###########################

os_client_args = {
    "hosts": [{"host": "localhost", "port": 9200}],
    "http_compress": True,
    "http_auth": ('admin', 'admin'),
    "use_ssl": True,
    "verify_certs": False,
    "ssl_assert_hostname": False,
    "ssl_show_warn": False,
    "timeout": 120
}

index_settings = {
    "body": {
        "settings": {
            "index.knn": True,
            "number_of_shards": 5,
            "number_of_replicas": 1
        },
        "mappings": {
            "properties": {
                "embedding": {
                  "dimension": 768,
                  "method": {
                    "engine": "faiss",
                    "space_type": "l2",
                    "name": "hnsw",
                    "parameters": {}
                  },
                  "type": "knn_vector"
                },
            }
        }
    }
}

index = "textract-mpnet"
s3_path = "s3://aryn-datasets-us-east-1/financebench/pdfs/"
manifest_path = "s3://aryn-datasets-us-east-1/financebench/manifest.json"
manifest_path_local="/home/admin/manifest_s3.json"

# hf_model = "sentence-transformers/all-MiniLM-L6-v2"
hf_model = "sentence-transformers/all-mpnet-base-v2"
# hf_model = "Alibaba-NLP/gte-large-en-v1.5"
# hf_model = "FinLang/finance-embeddings-investopedia"

openai_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value)
tokenizer = HuggingFaceTokenizer(hf_model)

###########################

sample = """"
    TEXT REPRESENTATION: 'Table of Contents\n\n3M Company and Subsidiaries\nConsolidated Balance Sheet\nAt December 31\n\nThe accompanying Notes to Consolidated Financial Statements are an integral part of this statement.\n\n50\n'
    KEYWORDS: '3M Company, Balance Sheet'
    ______________________________________________________________________
    TEXT REPRESENTATION: '3M Company and Subsidiaries Consolidated Balance Sheet At December 31\n,0,1,2\n0,"(Dollars in millions, except per share amount) ",2022 ,2021 \n1,Assets ,,\n2,Current assets ,,\n3,Cash and cash equivalents ,"$ 3,655 ","$ 4,564 "\n4,Marketable securities - current ,238 ,201 \n5,Accounts receivable - net of allowances of$174 and $189 ,"4,532 ","4,660 "\n6,Inventories ,,\n7,Finished goods ,"2,497 ","2,196 "\n8,Work in process ,"1,606 ","1,577 "\n9,Raw materials and supplies ,"1,269 ","1,212 "\n10,Total inventories ,"5,372 ","4,985 "\n11,Prepaids ,435 ,654 \n12,Other current assets ,456 ,339 \n13,Total current assets ,"14,688 ","15,403 "\n14,"Property, plant and equipment ","25,998 ","27,213 "\n15,Less: Accumulated depreciation ,"(16,820) ","(17,784) "\n16,"Property, plant and equipment - net ","9,178 ","9,429 "\n17,Operating lease right of use assets ,829 ,858 \n18,Goodwill ,"12,790 ","13,486 "\n19,Intangible assets - net ,"4,699 ","5,288 "\n20,Other assets ,"4,271 ","2,608 "\n21,Total assets ,"$ 46,455 ","$ 47,072 "\n22,Liabilities ,,\n23,Current liabilities ,,\n24,Short-term borrowings and current portion of long-term debt ,"$ 1,938 ","$ 1,307 "\n25,Accounts payable ,"3,183 ","2,994 "\n26,Accrued payroll ,692 ,"1,020 "\n27,Accrued income taxes ,259 ,260 \n28,Operating lease liabilities - current ,261 ,263 \n29,Other current liabilities ,"3,190 ","3,191 "\n30,Total current liabilities ,"9,523 ","9,035 "\n31,Long-term debt ,"14,001 ","16,056 "\n32,Pension and postretirement benefits ,"1,966 ","2,870 "\n33,Operating lease liabilities ,580 ,591 \n34,Other liabilities ,"5,615 ","3,403 "\n35,Total liabilities ,"31,685 ","31,955 "\n36,Commitments and contingencies (Note 16) ,,\n37,Equity ,,\n38,3M Company shareholders\' equity: ,,\n39,"Common stock par value, $.01 par value; 944,033,056 shares issued ",9 ,9 \n40,"Shares outstanding - December 31, 2022: 549,245,105 ",,\n41,"Shares outstanding - December 31, 2021: 571,845,478 ",,\n42,Additional paid-in capital ,"6,691 ","6,429 "\n43,Retained earnings ,"47,950 ","45,821 "\n44,"Treasury stock, at cost: ","(33,255) ","(30,463) "\n45,Accumulated other comprehensive income (loss) ,"(6,673) ","(6,750) "\n46,Total 3M Company shareholders\' equity ,"14,722 ","15,046 "\n47,Noncontrolling interest ,48 ,71 \n48,Total equity ,"14,770 ","15,117 "\n49,Total liabilities and equity ,"$ 46,455 ","$ 47,072 "\n\nThe accompanying Notes to Consolidated Financial Statements are an integral part of this statement.\n'
    KEYWORDS: 'Balance Sheet, Assets, Liabilities, Equity'
    ______________________________________________________________________
    TEXT REPRESENTATION: "Note 5 - Acquisitions and Divestitures\n\nNote 5 - Acquisitions and Divestitures\n\nYear ended June 30, 2023\n\nAcquisitions\n\n    On August 1, 2022, the Company completed the acquisition of 100% equity interest in a Czech Republic company that operates a world-class\nflexible packaging manufacturing plant. The purchase consideration of $59 million included a deferred portion of $5 million that was paid in the\nfirst quarter of fiscal year 2024. The acquisition is part of the Company's Flexibles reportable segment and resulted in the recognition of acquired\nidentifiable net assets of $36 million and goodwill of $23 million. Goodwill is not deductible for tax purposes. The fair values of the identifiable\nnet assets acquired and goodwill are based on the Company's best estimate as of June 30, 2023.\n\n    On March 17, 2023, the Company completed the acquisition of 100% equity interest in a medical device packaging manufacturing site in\nShanghai, China. The purchase consideration of $60 million is subject to customary post-closing adjustments. The consideration includes\ncontingent consideration of $20 million, to be earned and paid in cash over the three years following the acquisition date, subject to meeting\ncertain performance targets. The acquisition is part of the Company's Flexibles reportable segment and resulted in the recognition of acquired\nidentifiable net assets of $21 million and goodwill of $39 million. Goodwill is not deductible for tax purposes. The fair values of the contingent\nconsideration, identifiable net assets acquired, and goodwill are based on the Company's best estimate as of June 30, 2023, and are considered\npreliminary. The Company aims to complete the purchase price allocation as soon as practicable but no later than one year from the date of the\nacquisition.\n"
    KEYWORDS: Acquisitions and Divestitures, Czech Republic company, manufacturing site in\nShanghai, China
    ______________________________________________________________________
    TEXT REPRESENTATION: "    On May 31, 2023, the Company completed the acquisition of a New Zealand based leading manufacturer of state-of-the-art, automated protein\npackaging machines. The purchase consideration of $45 million is subject to customary post-closing adjustments. The consideration includes\ncontingent consideration of $13 million, to be earned and paid in cash over the two years following the acquisition date, subject to meeting\ncertain performance targets. The acquisition is part of the Company's Flexibles reportable segment and resulted in the recognition of acquired\nidentifiable net assets of $9 million and goodwill of $36 million. Goodwill is deductible for tax purposes. The fair values of the contingent\nconsideration, identifiable net assets acquired, and goodwill are based on the Company's best estimate as of June 30, 2023, and are considered\npreliminary. The Company aims to complete the purchase price allocation as soon as practicable but no later than one year from the date of the\nacquisition.\n\n    The fair value estimates for all three acquisitions were based on income, market, and cost valuation methods. Pro forma information related to\nthese acquisitions has not been presented, as the effect of the acquisitions on the Company's consolidated financial statements was not material.\n\nDisposal of Russian business\n\n    On December 23, 2022, the Company completed the sale of its Russian business after receiving all necessary regulatory approvals and cash\nproceeds, including receipt of closing cash balances. The sale follows the Company’s previously announced plan to pursue the orderly sale of its\nRussian business. The total net cash consideration received, excluding disposed cash and items settled net, was $365 million and resulted in a\npre-tax net gain of $215 million. The carrying value of the Russian business had previously been impaired by $90 million in the quarter ended\nJune 30, 2022. The impairment charge was based on the Company's best estimate of the fair value of its Russian business, which considered the\nwide range of indicative bids received and uncertain regulatory environment. The net pre-tax gain on disposal of the Russian business has been\nrecorded as restructuring, impairment, and other related activities, net within the consolidated statements of income. The Russian business had a\nnet carrying value of $252 million, including allocated goodwill of $46 million and accumulated other comprehensive losses of $73 million,\nprimarily attributed to foreign currency translation adjustments.\n\nYear ended June 30, 2022\n"
    KEYWORDS: acquisition, New Zealand based leading manufacturer, Disposal of Russian business

"""

class KeywordExtractorZeroShotGuidancePrompt(SimpleGuidancePrompt):
    system = "You are a helpful keyword extractor."
    user = """You are given a text representation of an element in a document. You need to extract up to 6 keywords from this text. Here are some examples of keyword extraction from
    text representations of elements:
    {examples}
    Using the text representation from the document and the provided examples, FIND, COPY, and RETURN the keywords. Return only a series of 6 words/phrases separated by
    commas as your answer. DO NOT REPHRASE OR MAKE UP AN ANSWER. This is the text representation:
    {text}
    """

prompt = KeywordExtractorZeroShotGuidancePrompt()

def keyword_enhance(elem):
    # llm call: give it a prompt, example, and text representation of element
    # add the result of the llm call to the element under a new 'keyword' field
    keywords = openai_llm.generate(
        prompt_kwargs={
            "prompt": prompt,
            "examples": sample,
            "text": elem.text_representation,
        }
    )
    elem.properties.update({"keywords": keywords})
    return elem

# TODO@aanya: redo with prefix & table description [no captions, how much will table description help?]

###########################

start = time()
embedder = SentenceTransformerEmbedder(model_name=hf_model, batch_size=100)
# embedder = OpenAIEmbedder(batch_size=100)
# table_extractor=TextractTableExtractor(profile_name="AdministratorAccess-237550789389", region_name="us-east-1", s3_upload_root="s3://aanya-textract")
table_extractor=CachedTextractTableExtractor(region_name="us-east-1", s3_cache_location="s3://aanya-textract/cache", s3_textract_upload_path="s3://aanya-textract/upload")
ds_list = []

# for i in [[0,15],[15,30],[30,45],[45,60],[60,75]]:
for i in [[70,75]]:
    ctx = sycamore.init()
    reader = ManifestReader(ctx)
    ds = (
        reader.binary(binary_format="pdf", metadata_provider=JsonManifestMetadataProvider(manifest_path_local), filesystem=get_s3_fs(), file_range=i)
        .partition(partitioner=SycamorePartitioner(extract_table_structure=False, threshold=0.35, use_ocr=False), table_extractor=table_extractor, num_gpus=0.1, compute=ActorPoolStrategy(size=1))
        .regex_replace(COALESCE_WHITESPACE)
        .merge(merger=GreedyTextElementMerger(tokenizer, 512))
        # .map_elements(keyword_enhance)
        .spread_properties(["path", "company", "year", "doc-type"])
        .explode()
        .embed(embedder=embedder, num_gpus=0.1)
    )
    ds_list.append(ds)

end = time()
print(f"Took {(end - start) / 60} mins")

###########################

for ds in ds_list:
    start = time()

    ds.write.opensearch(
        os_client_args=os_client_args,
        index_name=index,
        index_settings=index_settings,
    )
    # print (ds.take_all()[0].elements)

    end = time()
    print(f"Took {(end - start) / 60} mins")