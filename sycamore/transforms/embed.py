import json
import logging
import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Callable

import openai
import ray
from ray.data import ActorPoolStrategy, Dataset
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random, retry_if_exception_type

from sycamore.data import Document
from sycamore.llms.llms import OpenAIClientParameters
from sycamore.plan_nodes import Node, Transform
from sycamore.utils import batched, generate_map_batch_function, generate_map_batch_class_from_callable

logger = logging.getLogger(__name__)


def _pre_process_document(document: Document) -> str:
    return document.text_representation if document.text_representation is not None else ""


class Embedder(ABC):
    def __init__(
        self,
        model_name: str,
        batch_size: Optional[int] = None,
        model_batch_size: int = 100,
        pre_process_document: Optional[Callable[[Document], str]] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model_batch_size = model_batch_size
        self.pre_process_document = pre_process_document if pre_process_document else _pre_process_document

        if device is None:
            import torch.cuda

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def __call__(self, doc_batch: list[Document]) -> list[Document]:
        return self.generate_embeddings(doc_batch)

    @abstractmethod
    def generate_embeddings(self, doc_batch: list[Document]) -> list[Document]:
        pass


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

    def __init__(
        self,
        model_name: str,
        batch_size: Optional[int] = None,
        model_batch_size: int = 100,
        pre_process_document: Optional[Callable[[Document], str]] = None,
        device: Optional[str] = None,
    ):
        super().__init__(model_name, batch_size, model_batch_size, pre_process_document, device)
        self.type = type
        self._transformer: Optional[SentenceTransformer] = None

    def generate_embeddings(self, doc_batch: list[Document]) -> list[Document]:
        if not self._transformer:
            self._transformer = SentenceTransformer(self.model_name)

        assert self._transformer is not None

        text_batch = [self.pre_process_document(doc) for doc in doc_batch if doc.text_representation is not None]
        embeddings = self._transformer.encode(text_batch, batch_size=self.model_batch_size, device=self.device)

        i = 0
        for doc in doc_batch:
            if doc.text_representation is not None:
                doc.embedding = embeddings[i].tolist()
                i += 1

        return doc_batch


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
        model_name: str = OpenAIEmbeddingModels.TEXT_EMBEDDING_ADA_002.value,
        batch_size: Optional[int] = None,
        model_batch_size: int = 100,
        pre_process_document: Optional[Callable[[Document], str]] = None,
        api_key: Optional[str] = None,
        params: OpenAIClientParameters = OpenAIClientParameters(),
    ):
        super().__init__(model_name, batch_size, model_batch_size, pre_process_document, device="cpu")

        self._params = params

        if api_key is not None:
            self._params.api_key = api_key

        assert self._params.api_key is not None, (
            "You must provide an API key to "
            "use the LLM. Either pass it in "
            "the constructor or set the "
            "OPENAI_API_KEY environment "
            "variable."
        )

        self.model_name = model_name

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random(min=1, max=2),
        retry=retry_if_exception_type(openai.error.RateLimitError),
    )
    def _openai_embeddings(self, text_to_embed: list[str]) -> list:
        return openai.Embedding.create(**self._params.merge(self.model_name, input=text_to_embed)).data

    def generate_embeddings(self, doc_batch: list[Document]) -> list[Document]:
        # TODO: Add some input validation here.
        # The OpenAI docs are quite vague on acceptable values for model_batch_size.

        if self._params.is_azure() and self.model_batch_size > 16:
            logger.warn("The maximum batch size for emeddings on Azure Open AI is 16.")
            self.model_batch_size = 16

        for batch in batched(doc_batch, self.model_batch_size):
            text_to_embed = [
                self.pre_process_document(doc).replace("\n", " ")
                for doc in batch
                if doc.text_representation is not None
            ]

            embeddings = self._openai_embeddings(text_to_embed)

            i = 0
            for doc in batch:
                if doc.text_representation is not None:
                    doc.embedding = embeddings[i].embedding
                    i += 1

        return doc_batch


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
        pre_process_document: Optional[Callable[[Document], str]] = None,
        boto_session_args: list[Any] = [],
        boto_session_kwargs: dict[str, Any] = {},
    ):
        # Bedrock embedding curently doesn't support batching
        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            model_batch_size=1,
            pre_process_document=pre_process_document,
            device="cpu",
        )
        self.boto_session_args = boto_session_args
        self.boto_session_kwargs = boto_session_kwargs

    def _generate_embedding(self, client, text: str) -> list[float]:
        response = client.invoke_model(
            body=json.dumps({"inputText": text.replace("\n", " ")}),
            modelId=self.model_name,
            accept="application/json",
            contentType="application/json",
        )
        body_dict = json.loads(response.get("body").read())
        return body_dict["embedding"]

    def generate_embeddings(self, doc_batch: list[Document]) -> list[Document]:
        import boto3

        boto3.session.Session(*self.boto_session_args, **self.boto_session_kwargs)
        client = boto3.client("bedrock-runtime")

        for doc in doc_batch:
            if doc.text_representation is not None:
                doc.embedding = self._generate_embedding(client, self.pre_process_document(doc))
        return doc_batch


class Embed(Transform):
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
        super().__init__(child, **resource_args)
        self._embedder = embedder

    def execute(self) -> Dataset:
        dataset = self.child().execute()
        if self._embedder.device == "cuda":
            available_gpus = ray.available_resources().get("GPU")
            if "num_gpus" not in self.resource_args:
                self.resource_args["num_gpus"] = 1
            if self.resource_args["num_gpus"] <= 0:
                raise RuntimeError("Invalid GPU Nums!")
            gpu_per_task = self.resource_args["num_gpus"]

            output = dataset.map_batches(
                generate_map_batch_class_from_callable(self._embedder.generate_embeddings),
                batch_size=self._embedder.batch_size,
                compute=ActorPoolStrategy(min_size=1, max_size=math.ceil(available_gpus / gpu_per_task)),
                **self.resource_args
            )
        else:
            # in case of no gpu required, we use tasks to make it easier
            # to be fusible
            output = dataset.map_batches(
                generate_map_batch_function(self._embedder.generate_embeddings),
                batch_size=self._embedder.batch_size,
                **self.resource_args
            )

        return output
