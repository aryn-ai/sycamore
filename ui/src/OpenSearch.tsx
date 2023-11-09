const host = "localhost";
const protocol = "http";
const port = "3001/opensearch";
const auth = "admin:admin";

const SOURCES = ["type", "_id", "doc_id", "properties", "title", "text_representation", "bbox"]
// const SEARCH_PIPELINE = "ga-demo-pipeline-hybrid"
const SEARCH_PIPELINE = "hybrid_rag_pipeline"
const NO_RAG_SEARCH_PIPELINE = "hybrid_pipeline"

export const hybridConversationSearchNoRag = async (rephrasedQuestion: string, index_name: string, model_id: string) => {
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
                                "model_id": model_id
                            }
                        }
                    }
                ]
            }
        },
        "size": 20
    }
    const url = protocol + "://" + host + ":" + port + "/" + index_name + "/_search?search_pipeline=" + NO_RAG_SEARCH_PIPELINE
    return openSearchCall(query, url)
}

export const hybridConversationSearch = async (question: string, rephrasedQuestion: string, conversationId: string, index_name: string, embeddingModel: string, llmModel: string, numDocs: number = 7) => {
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
                                "model_id": embeddingModel
                            }
                        }
                    }
                ]
            }
        },
        "ext": {
            "generative_qa_parameters": {
                "llm_question": question,
                "conversation_id": conversationId,
                "context_size": numDocs,
                "llm_model": llmModel,
            }
        },
        "size": 20
    }
    const url = protocol + "://" + host + ":" + port + "/" + index_name + "/_search?search_pipeline=" + SEARCH_PIPELINE
    return openSearchCall(query, url)
}
export const getIndices = async () => {
    const url = protocol + "://" + host + ":" + port + "/_aliases?pretty"
    return openSearchNoBodyCall(url)
}
export const getEmbeddingModels = async () => {
    const body = {
      "query": {
        "bool": {
          "must_not": {
            "range": {
              "chunk_number": {
                "gte": 0
              } 
            } 
          },
          "must": [
            { "term": {"algorithm": "TEXT_EMBEDDING"}},
            { "term": {"model_state": "DEPLOYED"}}
          ]
        }
      }
    }
    const url = protocol + "://" + host + ":" + port + "/_plugins/_ml/models/_search"
    return openSearchCall(body, url)
}
export const createConversation = async (conversationId: string) => {
    const body = {
        "name": conversationId
    }
    const url = protocol + "://" + host + ":" + port + "/_plugins/_ml/memory/conversation"
    return openSearchCall(body, url)
}
export const getInteractions = async (conversationId: any) => {
    const url = protocol + "://" + host + ":" + port + "/_plugins/_ml/memory/conversation/" + conversationId
    return openSearchNoBodyCall(url)
}
export const getConversations = () => {
    const url = protocol + "://" + host + ":" + port + "/_plugins/_ml/memory/conversation/"
    return openSearchNoBodyCall(url)
}
export const deleteConversation = async (conversation_id: string) => {
    // hack for empty conversation delete:
    console.log("Going to delete", conversation_id)
    const url = protocol + "://" + host + ":" + port + "/_plugins/_ml/memory/conversation/" + conversation_id

    const body = {
        input: "",
        prompt_template: "",
        response: ""
    }
    console.log("Adding interaction")
    const addCall = await openSearchCall(body, url)
    await addCall

    console.log("Now deleting")
    return openSearchNoBodyCall(url, "DELETE")
}
export const openSearchNoBodyCall = async (url: string, http_method: string = "GET") => {
    try {
        console.log("sending ", http_method + " " + url)
        const response = await fetch(url, {
            method: http_method,
            headers: {
                'Content-Type': 'application/json',
            },
        });
        if (!response.ok) {
            return response.text().then(text => {
                throw new Error(`OpenSearchRequest rejected with status ${response.status} and message ${text}`);
            })
        }

        const data = await response.json();
        console.log('Response data:', data);
        return data;
    } catch (error: any) {
        console.error('Error sending query:', error);
        throw new Error("Error making OpenSearch query to " + url + " without body: " + error.message);
    }
}
export const openSearchCall = async (query: any, url: string, http_method: string = "POST") => {
    try {
        console.log("sending request", url, JSON.stringify(query))
        const response = await fetch(url, {
            method: http_method,
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(query),
        });
        if (!response.ok) {
            return response.text().then(text => {
                throw new Error(`Request to ${url}:\n` + JSON.stringify(query) + `\nrejected with status ${response.status} and message ${text}`);
            })
        }

        const data = await response.json();
        console.log('Response data:', data);
        return data;
    } catch (error: any) {
        console.error('Error sending query:', error);
        throw new Error("Error making OpenSearch to " + url + " query with body: " + error.message);
    }
}


// Legacy for local RAG
export const sendQuery = async (query: any, index_name: string) => {
    const url = protocol + "://" + host + ":" + port + "/" + index_name + `/_search/`;
    return openSearchCall(query, url)
}

export async function queryOpenSearch(question: string, index_name: string, model_id: string) {
    const query = {
        "query": {
            "bool": {
                "should": [
                    {
                        "function_score": {
                            "query": {
                                "match": {
                                    "text_representation": question
                                }
                            },
                            "weight": 0.925
                        }
                    },
                    {
                        "neural": {
                            "embedding": {
                                "query_text": question,
                                "model_id": model_id,
                                "k": 20
                            }
                        }
                    }
                ]
            }
        },
        "size": 10,
        "_source": SOURCES
    }
    console.log("OS question ", question)
    console.log("OS query ", JSON.stringify(query))
    var response = sendQuery(query, index_name);
    return response;
}