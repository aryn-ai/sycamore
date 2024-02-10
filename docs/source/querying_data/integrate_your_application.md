# Integrating a chat UI with Sycamore

The UI is a key component in many conversational search applications. A good UI enables users to search for data and see natural langauge answers and relevant search results in an easy to use interface. Additionally, UIs can show the source information for natural language answers. This section provides an overview of how you can create a simple web application that uses Sycamore's APIs for conversational search. The application will be built using React, TypeScript, and OpenAI.

## Prerequisites

This tutorial requires a running Sycamore stack. You can follow the steps in the [get started guide](../welcome_to_sycamore/get_started.md) to set it up. The specific components and steps are:

1. Running Sycamore stack
2. Ingest the Sort Benchmark data into your stack

## Building the components

### Chat sessions and conversation memory

One of the benefits of building apps with Sycamore's APIs is how simple it is to manage client sessions and conversations. The API provides CRUD operations on a conversation object, which allows you to remotely store conversation state and history while not having to worry about client side memory.

Each user session can be modeled as a new `conversation` in Sycamore. Additionally, the user can pick an existing conversation and continue from where it was left off - retaining all of the context that was used so far. A conversation’s identifier is used each time the user asks a question, so that interaction is automatically added to that conversation's history.

You access conversation memory through Sycamore's REST API, and it's easy to perform conversation level operations. Here is an example about how you can create a conversation in TypeScript

```typescript
const body = {
  "name": <user defined name>
}
const url = protocol + "://" + host + ":" + port + "/_plugins/_ml/memory/conversation"

try {
  const response = await fetch(url, {
      method: "POST",
      headers: {
          'Content-Type': 'application/json',
      },
      body: JSON.stringify(query),
  });
  if (!response.ok) {
      return response.text().then(text => {
          throw new Error(`Request rejected with status ${response.status} and message ${text}`);
      })
  }
  const data = await response.json();
  return data;
} catch (error: any) {
  console.error('Error sending query:', error);
  throw new Error("Error making OpenSearch query: " + error.message);
}
```

Similarly, you can list existing conversations with a GET request:

```typescript
const url = protocol + "://" + host + ":" + port + "/_plugins/_ml/memory/conversation"
try {
    const response = await fetch(url, {
        method: "GET",
        headers: {
            'Content-Type': 'application/json',
        },
    });
    if (!response.ok) {
        return response.text().then(text => {
            throw new Error(`Request rejected with status ${response.status} and message ${text}`);
        })
    }
    const data = await response.json();
    return data;
} catch (error: any) {
    console.error('Error sending query:', error);
    throw new Error("Error making OpenSearch query: " + error.message);
}
```

> Note: By default, all conversations are visible to every client accessing the cluster. In a production setup, you will want to [use access control](https://opensearch.org/docs/latest/security/access-control/index/) to restrict the conversations a user can see.

### Queries and interactions

Now that we have initialized a conversation, we can move on to processing user questions and returning answers. Our application will do this through invoking the [RAG Search Pipeline](../querying_data/using_rag_pipelines.md).

To perform a conversational search request with RAG, see the following example

```typescript
const SOURCES = ["type", "_id", "doc_id", "properties", "title", "text_representation"]
const MODEL_ID = "<your neural search model>"
const SEARCH_PIPELINE = "hybrid_rag_pipeline"
const LLM_MODEL = "gpt4"

const userQuestion = "Who created them?"
const rephrasedQuestion = "Who created the sort benchmarks?"
const conversationId = "[active conversation id]"

const query =
{
  "_source": SOURCES,
  "query": {
      "hybrid": {
          "queries": [
              {
                  "match": {
                      "text_representation": rephrasedQuestion
                  }
              },
              {
                  "neural": {
                      "embedding": {
                          "query_text": rephrasedQuestion,
                          "k": 100,
                          "model_id": MODEL_ID
                      }
                  }
              }
          ]
      }
  },
  "ext": {
      "generative_qa_parameters": {
          "llm_question": userQuestion,
          "conversation_id": conversationId,
          "llm_model": LLM_MODEL,
      }
  },
  "size": 20
}
const url = protocol + "://" + host + ":" + port + "/" + index_name + "/_search?search_pipeline=" + SEARCH_PIPELINE

try {
  const response = await fetch(url, {
      method: "POST",
      headers: {
          'Content-Type': 'application/json',
      },
      body: JSON.stringify(query),
  });
  if (!response.ok) {
      return response.text().then(text => {
          throw new Error(`Request rejected with status ${response.status} and message ${text}`);
      })
  }

  const data = await response.json();
  return data;
} catch (error: any) {
  console.error('Error sending query:', error);
  throw new Error("Error making OpenSearch query: " + error.message);
}
```

### Document highlighting

Certain documents, like PDFs, will contain additional metadata about what section of the documents were used to generate a response.

For a PDF search result, the document contains a `properties` attribute, that will optionally contains `boxes`. Each box represents a page number, and the 4 coordinates of a bounding box within that page that represent the text, image, or table that was used as data. You can use a library like `react-pdf` to visualize this client side. Your component might look like this:

```typescript
<Document file={url} onLoadSuccess={onDocumentLoadSuccess}>
    <Page pageNumber={pageNumber}>
        {boxes[pageNumber] && boxes[pageNumber].map((box: any, index: number) => (
            <div
                key={index}
                style={{
                    position: "absolute",
                    backgroundColor: "#ffff0033",
                    left: `${box[0] * 100}%`,
                    top: `${box[1] * 100}%`,
                    width: `${(box[2] - box[0]) * 100}%`,
                    height: `${(box[3] - box[1]) * 100}%`,
                }} />
        ))}
    </Page>
</Document>
```

### Question rewriting

A key component of conversational search is to use context during conversation to improve the way the system understands the question submitted by the user. For instance, this allows a user to ask a question like “when was it created?” because they may have previously asked “what is JouleSort?”, and the system will know “it” refers to “JouleSort” from the conversational context.

We can easily implement this in our application by passing the interactions of our `conversation` to an LLM and ask it to rephrase the question. An example can look like:

```typescript
export const generate_question_rewriting_prompt = (text: string, conversation: any()) => {
  const conversation = conversation.interactions.map((interaction) => {
      return [
        { role: "user", content: interaction.input },
        { role: "system", content: interaction.response }
      ]
  })
  const sys = "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. \n"
  const prompt = "Follow Up Input: " + text + "\nStandalone question: "
  const openAiPrompt = [{ role: "system", content: sys }, ...conversation, { role: "user", content: prompt }]
  return openAiPrompt;
}
const conversation = [GET conversation result]
const prompt = generate_question_rewriting_prompt("when was it created?", conversation)
// make OpenAI request with $prompt
```

## Conclusion

This tutorial showed how you can use Sycamore’s conversational APIs to easily implement the core components of a client side conversational search application. For more details about the conversation APIs leveraged by Sycamore, see the [OpenSearch documentation](https://opensearch.org/docs/latest/ml-commons-plugin/conversational-search/).
