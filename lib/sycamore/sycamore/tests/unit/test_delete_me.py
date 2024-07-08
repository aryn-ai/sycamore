import pickle

from sycamore.llms import OpenAI
from sycamore.utils.cache import S3Cache


def testa():
    cache = S3Cache("s3://bucket/key")
    llm = OpenAI("gpt-3.5-turbo", cache=cache)
    pickled_obj = pickle.dumps(llm)
    pickle.loads(pickled_obj)
