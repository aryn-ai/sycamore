import json
from unittest.mock import patch
import tempfile

from sycamore.llms.prompts import RenderedPrompt, RenderedMessage
from sycamore.llms.bedrock import Bedrock, BedrockModels, DEFAULT_ANTHROPIC_VERSION, DEFAULT_MAX_TOKENS
from sycamore.utils.cache import DiskCache


class BedrockBody:
    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body


def bedrock_reply(body):
    return {
        "ResponseMetadata": {
            "HTTPStatusCode": 200,
            "HTTPHeaders": {
                "x-amzn-bedrock-invocation-latency": 1111,
                "x-amzn-bedrock-input-token-count": 30,
                "x-amzn-bedrock-output-token-count": 50,
            },
        },
        "body": BedrockBody(body),
    }


@patch("boto3.client")
def test_bedrock_simple(mock_boto3_client):
    mock_boto3_client.return_value.invoke_model.return_value = bedrock_reply(
        '{ "content": [{"text": "Here is your result: 56"}]}'
    )

    client = Bedrock(BedrockModels.CLAUDE_3_5_SONNET)
    assert client.is_chat_mode()
    assert client._model_name == BedrockModels.CLAUDE_3_5_SONNET.value.name

    result = client.generate(
        prompt=RenderedPrompt(
            messages=[
                RenderedMessage(role="user", content="Roll 4d20 and tell me the final sum."),
            ]
        )
    )
    assert result == "Here is your result: 56"

    assert mock_boto3_client.call_args.kwargs["service_name"] == "bedrock-runtime"
    assert json.loads(mock_boto3_client.return_value.invoke_model.call_args.kwargs["body"]) == {
        "messages": [{"role": "user", "content": "Roll 4d20 and tell me the final sum."}],
        "max_tokens": DEFAULT_MAX_TOKENS,
        "anthropic_version": DEFAULT_ANTHROPIC_VERSION,
        "temperature": 0,
    }


@patch("boto3.client")
def test_bedrock_system_role(mock_boto3_client):
    mock_boto3_client.return_value.invoke_model.return_value = bedrock_reply(
        '{"content": [{"text": "Here is your result: 56"}]}'
    )

    client = Bedrock(BedrockModels.CLAUDE_3_5_SONNET)
    assert client.is_chat_mode()
    assert client._model_name == BedrockModels.CLAUDE_3_5_SONNET.value.name

    result = client.generate(
        prompt=RenderedPrompt(
            messages=[
                RenderedMessage(role="system", content="You are a DM for a game of D&D."),
                RenderedMessage(role="user", content="Roll 4d20 and tell me the final sum."),
            ]
        )
    )
    assert result == "Here is your result: 56"

    assert mock_boto3_client.call_args.kwargs["service_name"] == "bedrock-runtime"
    assert json.loads(mock_boto3_client.return_value.invoke_model.call_args.kwargs["body"]) == {
        "messages": [
            {"role": "user", "content": "You are a DM for a game of D&D.\nRoll 4d20 and tell me the final sum."}
        ],
        "max_tokens": DEFAULT_MAX_TOKENS,
        "anthropic_version": DEFAULT_ANTHROPIC_VERSION,
        "temperature": 0,
    }


@patch("boto3.client")
def test_bedrock_with_llm_kwargs(mock_boto3_client):
    mock_boto3_client.return_value.invoke_model.return_value = bedrock_reply(
        '{"content": [{"text": "Here is your result: 56"}]}'
    )

    client = Bedrock(BedrockModels.CLAUDE_3_5_SONNET)
    assert client.is_chat_mode()
    assert client._model_name == BedrockModels.CLAUDE_3_5_SONNET.value.name

    result = client.generate(
        prompt=RenderedPrompt(
            messages=[
                RenderedMessage(role="user", content="Roll 4d20 and tell me the final sum."),
            ]
        ),
        llm_kwargs={"max_tokens": 100, "anthropic_version": "v1"},
    )
    assert result == "Here is your result: 56"

    assert mock_boto3_client.call_args.kwargs["service_name"] == "bedrock-runtime"
    assert json.loads(mock_boto3_client.return_value.invoke_model.call_args.kwargs["body"]) == {
        "messages": [{"role": "user", "content": "Roll 4d20 and tell me the final sum."}],
        "max_tokens": 100,
        "anthropic_version": "v1",
        "temperature": 0,
    }


@patch("boto3.client")
def test_bedrock_with_cache(mock_boto3_client):
    mock_boto3_client.return_value.invoke_model.return_value = bedrock_reply(
        '{"content": [{"text": "Here is your result: 56"}]}'
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        cache = DiskCache(temp_dir)

        assert cache.cache_hits == 0
        assert cache.total_accesses == 0

        client = Bedrock(BedrockModels.CLAUDE_3_5_SONNET, cache=cache)
        assert client.is_chat_mode()
        assert client._model_name == BedrockModels.CLAUDE_3_5_SONNET.value.name

        result = client.generate(
            prompt=RenderedPrompt(
                messages=[
                    RenderedMessage(role="user", content="Roll 4d20 and tell me the final sum."),
                ]
            )
        )
        assert result == "Here is your result: 56"

        assert cache.cache_hits == 0
        assert cache.total_accesses == 1

        assert mock_boto3_client.call_args.kwargs["service_name"] == "bedrock-runtime"
        assert json.loads(mock_boto3_client.return_value.invoke_model.call_args.kwargs["body"]) == {
            "messages": [{"role": "user", "content": "Roll 4d20 and tell me the final sum."}],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "anthropic_version": DEFAULT_ANTHROPIC_VERSION,
            "temperature": 0,
        }

        result = client.generate(
            prompt=RenderedPrompt(
                messages=[
                    RenderedMessage(role="user", content="Roll 4d20 and tell me the final sum."),
                ]
            )
        )
        assert result == "Here is your result: 56"

        assert cache.cache_hits == 1
        assert cache.total_accesses == 2
