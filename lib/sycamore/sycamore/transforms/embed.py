import json
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Callable, Union, List
import logging

from openai import OpenAI as OpenAIClient
from openai import AzureOpenAI as AzureOpenAIClient

from sycamore.data import Document, Element
from sycamore.llms import OpenAIClientParameters
from sycamore.utils import choose_device

# from sycamore.llms.llms import AzureOpenAI, OpenAIClientParameters
from sycamore.llms.openai import OpenAIClientWrapper
from sycamore.plan_nodes import Node
from sycamore.transforms.map import MapBatch
from sycamore.utils import batched
from sycamore.utils.import_utils import requires_modules
from sycamore.utils.time_trace import timetrace


logger = logging.getLogger(__name__)


def _pre_process_document(document: Union[Document, Element]) -> str:
    return document.text_representation or ""


def _text_representation_is_empty(doc: Union[Document, Element]) -> bool:
    return doc.text_representation is None or doc.text_representation.strip() == ""


class Embedder(ABC):
    def __init__(
        self,
        model_name: str,
        batch_size: Optional[int] = None,
        model_batch_size: Optional[int] = None,
        pre_process_document: Optional[Callable[[Union[Document, Element]], str]] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.pre_process_document = pre_process_document if pre_process_document else _pre_process_document
        self.device = choose_device(device)
        self.model_batch_size = model_batch_size

    def __call__(self, doc_batch: list[Document]) -> list[Document]:
        return self.generate_embeddings(doc_batch)

    def generate_embeddings(self, doc_batch: list[Document]) -> list[Document]:
        """Handle batching and document processing logic in parent class"""

        # Collect objects to embed
        obj_for_embedding: list[Union[Document, Element]] = []
        text_to_embed = []

        # First pass: collect all texts that need embedding
        for doc in doc_batch:
            if not _text_representation_is_empty(doc):
                text_to_embed.append(self.pre_process_document(doc))
                obj_for_embedding.append(doc)

            if isinstance(doc, Document) and doc.get("elements"):
                for element in doc.elements:
                    if not _text_representation_is_empty(element):
                        text_to_embed.append(self.pre_process_document(element))
                        obj_for_embedding.append(element)

        # Return early if nothing to embed
        if not text_to_embed:
            return doc_batch

        # Generate embeddings
        all_embeddings = []
        for text_batch in batched(text_to_embed, self.model_batch_size):
            batch_embeddings = self.embed_texts(text_batch)
            all_embeddings.extend(batch_embeddings)

        # Assign embeddings
        for i, embedding in enumerate(all_embeddings):
            obj_for_embedding[i].embedding = embedding
        # assert embed_count == len(all_embeddings)
        return doc_batch

    @staticmethod
    def clamp_batch_size(batch_size, max_and_default=None):
        if batch_size < 1:
            raise ValueError(f"Batch size must be at least 1, got {batch_size}")
        if max_and_default is None:
            return batch_size
        if batch_size > max_and_default:
            logging.warning(
                f"Requested batch size {batch_size} exceeds maximum {max_and_default}. "
                f"Reducing to {max_and_default}."
            )
            return max_and_default
        return batch_size

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts. To be implemented by child classes."""
        pass

    def generate_text_embedding(self, text: str) -> list[float]:
        """Single text embedding wrapper"""
        return self.embed_texts([text])[0]


class SentenceTransformerEmbedder(Embedder):
    """
    SentenceTransformerEmbedder is an Embedder class for generating sentence embeddings using the
    SentenceTransformer model.

    Args:
        model_name: The name or path of the SentenceTransformer model to use for embedding.
        batch_size: The dataset batch size for embedding, if specified. Default is None.
        model_batch_size: The batch size used by the underlying SentenceTransformer model for embedding.
        device: The device (e.g., "cpu" or "cuda") on which to perform embedding.

    Example:
        .. code-block:: python

            model_name="sentence-transformers/all-MiniLM-L6-v2"
            embedder = SentenceTransformerEmbedder(batch_size=100, model_name=model_name)

            context = sycamore.init()
            pdf_docset = context.read.binary(paths, binary_format="pdf")
                .partition(partitioner=UnstructuredPdfPartitioner())
                .explode()
                .embed(embedder=embedder)



    """

    @requires_modules("sentence_transformers", extra="local-inference")
    def __init__(
        self,
        model_name: str,
        batch_size: Optional[int] = None,
        model_batch_size: int = 100,
        pre_process_document: Optional[Callable[[Union[Document, Element]], str]] = None,
        device: Optional[str] = None,
    ):
        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            model_batch_size=self.clamp_batch_size(model_batch_size),
            pre_process_document=pre_process_document,
            device=device,
        )
        self._transformer = None

    def _ensure_model(self):
        if not self._transformer:
            from sentence_transformers import SentenceTransformer

            self._transformer = SentenceTransformer(self.model_name)

    @timetrace("StEmbedder")
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        self._ensure_model()
        assert self._transformer is not None
        embeddings = self._transformer.encode(texts, batch_size=self.model_batch_size, device=self.device)
        return embeddings.tolist()


class OpenAIEmbeddingModels(Enum):
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"


class OpenAIEmbedder(Embedder):
    """Embedder implementation using the OpenAI embedding API.

    Args:
        model_name: The name of the OpenAI embedding model to use.
        batch_size: The Ray batch size.
        model_batch_size: The number of documents to send in a single OpenAI request.
    """

    def __init__(
        self,
        model_name: Union[str, OpenAIEmbeddingModels] = OpenAIEmbeddingModels.TEXT_EMBEDDING_ADA_002.value,
        batch_size: Optional[int] = None,
        model_batch_size: int = 100,
        pre_process_document: Optional[Callable[[Union[Document, Element]], str]] = None,
        api_key: Optional[str] = None,
        client_wrapper: Optional[OpenAIClientWrapper] = None,
        params: Optional[OpenAIClientParameters] = None,
        **kwargs,
    ):
        if isinstance(model_name, OpenAIEmbeddingModels):
            model_name = model_name.value

        if client_wrapper is None:
            if params is not None:
                client_wrapper = params
            else:
                if api_key is not None:
                    kwargs.update({"api_key": api_key})
                client_wrapper = OpenAIClientWrapper(**kwargs)
        else:
            if api_key is not None:
                client_wrapper.api_key = api_key

        self.client_wrapper = client_wrapper
        self._client: Optional[OpenAIClient] = None
        self.model_name = model_name

        client = client_wrapper.get_client()
        if isinstance(client, AzureOpenAIClient):
            default_batch_size = 16
        else:
            default_batch_size = None
        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            model_batch_size=self.clamp_batch_size(model_batch_size, default_batch_size),
            pre_process_document=pre_process_document,
            device="cpu",
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _ensure_client(self):
        if self._client is None:
            self._client = self.client_wrapper.get_client()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # TODO: Add some input validation here.
        # The OpenAI docs are quite vague on acceptable values for model_batch_size.
        self._ensure_client()
        assert self._client is not None
        response = self._client.embeddings.create(model=self.model_name, input=texts)
        return [data.embedding for data in response.data]


class BedrockEmbeddingModels(Enum):
    TITAN_EMBED_TEXT_V1 = "amazon.titan-embed-text-v1"


class BedrockEmbedder(Embedder):
    """Embedder implementation using Amazon Bedrock.

    Args:
        model_name: The Bedrock embedding model to use. Currently the only available
            model is amazon.titan-embed-text-v1
        batch_size: The Ray batch size.
        boto_session_args: Arg parameters to pass to the boto3.session.Session constructor.
            These will be used to create a boto3 session on each executor.
        boto_session_kwargs: Keyword arg parameters pass to the boto3.session.Session constructor.

    Example:
         .. code-block:: python

            embedder = BedrockEmbedder(boto_session_kwargs={'profile_name': 'my_profile'})
            docset_with_embeddings = docset.embed(embedder=embedder)
    """

    def __init__(
        self,
        model_name: str = BedrockEmbeddingModels.TITAN_EMBED_TEXT_V1.value,
        batch_size: Optional[int] = None,
        pre_process_document: Optional[Callable[[Union[Document, Element]], str]] = None,
        model_batch_size: int = 1,
        boto_session_args: list[Any] = [],
        boto_session_kwargs: dict[str, Any] = {},
    ):
        # Bedrock embedding curently doesn't support batching
        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            model_batch_size=self.clamp_batch_size(model_batch_size, 1),
            pre_process_document=pre_process_document,
            device="cpu",
        )
        self.boto_session_args = boto_session_args
        self.boto_session_kwargs = boto_session_kwargs
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import boto3

            boto3.session.Session(*self.boto_session_args, **self.boto_session_kwargs)
            self._client = boto3.client("bedrock-runtime")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        assert len(texts) == 1, "Bedrock only supports batch size 1"
        self._ensure_client()
        assert self._client is not None
        embeddings = []
        response = self._client.invoke_model(
            body=json.dumps({"inputText": texts[0].replace("\n", " ")}),
            modelId=self.model_name,
            accept="application/json",
            contentType="application/json",
        )
        body_dict = json.loads(response.get("body").read())
        embeddings.append(body_dict["embedding"])
        return embeddings


class Embed(MapBatch):
    """
    Embed is a transformation that generates embeddings a docset using an Embedder.

    The generated embeddings are stored in a special embedding property on each document.
    It utilizes an Embedder to perform the embedding process.

    Args:
        child: The source node or component that provides the dataset to be embedded.
        embedder: An instance of an Embedder class that defines the embedding method to be applied.
        resource_args: Additional resource-related arguments that can be passed to the embedding operation.

    Example:
         .. code-block:: python

            source_node = ...  # Define a source node or component that provides a dataset.
            custom_embedder = MyEmbedder(embedding_params)
            embed_transform = Embed(child=source_node, embedder=custom_embedder)
            embedded_dataset = embed_transform.execute()
    """

    def __init__(self, child: Node, embedder: Embedder, **resource_args):
        self.resource_args = resource_args
        if "batch_size" not in self.resource_args:
            self.resource_args["batch_size"] = embedder.batch_size

            # Batch size can be an integer, None, or the string "default" per
            # https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html
            batch_size = self.resource_args["batch_size"]
            assert (
                batch_size is None
                or (isinstance(batch_size, int) and batch_size > 0)
                or self.resource_args["batch_size"] == "default"
            )

        if embedder.device == "cuda":
            if "num_gpus" not in self.resource_args:
                self.resource_args["num_gpus"] = 1
            if self.resource_args["num_gpus"] <= 0:
                raise RuntimeError("Invalid GPU Nums!")
            if "parallelism" not in self.resource_args:
                self.resource_args["parallelism"] = 1
        elif embedder.device == "cpu":
            self.resource_args.pop("num_gpus", None)

        super().__init__(child, f=embedder, **resource_args)
