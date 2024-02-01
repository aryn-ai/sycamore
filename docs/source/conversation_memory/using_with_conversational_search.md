# Using with conversational search

You can use Sycamore's conversation memory to store the state of a conversation and use it for context for the next user interaction. This is critical when creating a chat experience. When using [Sycamore's RAG pipeline](../querying_data/using_rag_pipelines.md), you can easily specify a 'conversation_id' parameter in your request to add the interaction to that conversation. For example:


```javascript
POST /sort-benchmark/_search?search_pipeline=hybrd_rag_pipeline
{
    "_source": ["properties", "summary", "text_representation"],
    "query": {
        {
            "match": {
                "text_representation": "Who created the sort benchmarks?"
            }
        }
    },
    "size": 7,
    "ext": {
        "generative_qa_parameters": {
            "llm_question": "Who created the sort benchmarks?",
            "llm_model": "gpt-4",
            "conversation_id": "<conversation_id>"
        }
    }
}
```

The RAG pipeline will retrieve up to the last 10 of interactions from the conversation, add them to the LLM prompt, and then add this interaction (input/response pair) to the conversation. If you don't have a conversation, you can easily create one. If you don't specify a conversation ID, the interaction will not be added to a conversation and no prior context will be used in the LLM prompt.

## Question rewriting

We recommend using "question rewriting" when building a conversational search application, which also requires conversation memory. Question rewriting takes a user quesiton submitted to the application and changes it to reflect the context of the conversation. This can make the question more clear for Sycamore to return a better answer, especially if the question is vague or requires context from a prior interaction (e.g. using prepositions that refer to things in previous interactions). This also creates more of a natural conversational experience.

You can use the conversation memory APIs with the LLM of your choice (e.g. OpenAI), and ask the LLM to rephrase the user’s original question in the context of the specified conversation. The [Sycamore demo query UI](../querying_data/demo_query_ui.md) uses this process for question rewriting.

For example, say the user asked “Who created them?”

```
GET /_plugins/_ml/memory/conversation/<conversation_id>
```

Returns

```javascript
{
    "interactions": [
        {
            "conversation_id": "5Kf1uIoBk2CcrZQrUIec",
            "interaction_id": "5af2uIoBk2CcrZQreYeC",
            "create_time": "2023-09-21T18:17:55.280275368Z"
            "input": "What are the sort benchmarks?",
            "prompt_template": "Generate a concise and informative answer in...",
            "response": "Sort benchmarks are standardized tests used to evaluate the \
                        performance and efficiency of sorting algorithms and systems. \
                        They measure factors such as sort rate, cost efficiency, energy \
                        consumption, and overall system performance. Some common sort \
                        benchmarks include GraySort, CloudSort, MinuteSort, JouleSort, \
                        PennySort, and TeraByte Sort. These benchmarks provide a \
                        standardized framework for comparing different sorting solutions \
                        and determining their suitability for specific use cases.",
            "origin": "retrieval_augmented_generation",
            "additional_info": "[\"This passage is about the sort benchmarks\\n\\n \
                                    Top Results..."
        }
    ]
}
```

Then query OpenAI:

```bash
curl https://api.openai.com/v1/chat/completions \
        -H "Content-Type: application/json"\
        -H "Authorization: Bearer $OPENAI_API_KEY" \
-d'
{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "system",
            "content": "Rewrite the question taking into account the context from the previous several interactions"
        },
        {
            "role": "user",
            "content": "What are the sort benchmarks?"
        },
        {
            "role": "assistant",
            "content": "Sort benchmarks are standardized tests used to evaluate the \
                        performance and efficiency of sorting algorithms and systems. \
                        They measure factors such as sort rate, cost efficiency, energy \
                        consumption, and overall system performance. Some common sort \
                        benchmarks include GraySort, CloudSort, MinuteSort, JouleSort, \
                        PennySort, and TeraByte Sort. These benchmarks provide a \
                        standardized framework for comparing different sorting solutions \
                        and determining their suitability for specific use cases.",
        },
        {
            "role": "user",
            "content": "Question: Who created them? \n Rewritten Question:"
        }
    ],
    "temperature": 0.7
}'
```

Which responds:

```javascript
{
  "id": "chatcmpl-81Ki4wRPiEbYClpnxZBFqy5RacLvj",
  "object": "chat.completion",
  "created": 1695328628,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Who is responsible for creating and developing the sort benchmarks?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 134,
    "completion_tokens": 11,
    "total_tokens": 145
  }
}
```

This lets the RAG pipeline in OpenSearch understand the question the user is asking. The historical context has been ‘brought in’ to the rewritten query, so that part of the stack doesn't need to obtain the context from elsewhere. For example:

```javascript
POST /sort-benchmark/_search?search_pipeline=hybrid_rag_pipeline
{
    "_source": ["properties", "summary", "text_representation"],
    "query": {
        {
            "match": {
                "text_representation": "Who is responsible for creating and \
                                        developing the sort benchmarks?"
            }
        }
    },
    "size": 7,
    "ext": {
        "generative_qa_parameters": {
            "llm_question": "Who created them?",
            "llm_model": "gpt-3.5-turbo",
            "conversation_id": "<conversation_id>"
        }
    }
}
```
