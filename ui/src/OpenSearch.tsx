import semverGTE from "semver/functions/gte";
import {Mutex} from "async-mutex";

const SOURCES = ["type", "_id", "doc_id", "properties", "title", "text_representation", "bbox"]
// const SEARCH_PIPELINE = "ga-demo-pipeline-hybrid"
const SEARCH_PIPELINE = "hybrid_rag_pipeline"
const NO_RAG_SEARCH_PIPELINE = "hybrid_pipeline"
// access with getOpenSearchVersion
const OS_VERSION_OBJ: {osVersion: string} = {osVersion: ""}


const getOpenSearchVersion = async () => {
    if(OS_VERSION_OBJ.osVersion.length === 0) {
        const response = await fetch("/opensearch-version");
        if (response.ok) {
            OS_VERSION_OBJ.osVersion = await response.text();
            return OS_VERSION_OBJ.osVersion;
        } else {
            throw Error(await response.text())
        }
    } else {
        return OS_VERSION_OBJ.osVersion;
    }
}

export const is2dot12plus = async () => {
    const version = await getOpenSearchVersion();
    console.log("opensearch version: " + version)
    return semverGTE(version, "2.12.0");
}

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
    const url = "/opensearch/" + index_name + "/_search?search_pipeline=" + NO_RAG_SEARCH_PIPELINE
    return openSearchCall(query, url)
}

export const hybridConversationSearch = async (question: string, rephrasedQuestion: string, conversationId: string, index_name: string, embeddingModel: string, llmModel: string, numDocs: number = 7) => {
    let genQaParams: {[key: string]: any} = {
        "llm_question": question,
        "context_size": numDocs,
        "llm_model": llmModel,
    }
    if(await is2dot12plus()) {
        genQaParams.memory_id = conversationId;
    } else {
        genQaParams.conversation_id = conversationId;
    }
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
            "generative_qa_parameters": genQaParams
        },
        "size": 20
    }
    const url = "/opensearch/" + index_name + "/_search?search_pipeline=" + SEARCH_PIPELINE
    return openSearchCall(query, url)
}
export const getIndices = async () => {
    const url = "/opensearch/_aliases?pretty"
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
    const url = "/opensearch/_plugins/_ml/models/_search"
    return openSearchCall(body, url)
}
export const createConversation = async (conversationId: string) => {
    const body = {
        "name": conversationId
    }
    let url;
    if(await is2dot12plus()) {
        url = "/opensearch/_plugins/_ml/memory/"
    } else {
        url = "/opensearch/_plugins/_ml/memory/conversation"
    }
    return openSearchCall(body, url)
}
export const getInteractions = async (conversationId: any) => {
    if(await is2dot12plus()) {
        const url = "/opensearch/_plugins/_ml/memory/" + conversationId + "/_search"
        const body = {
            "query": {
                "match_all": {}
            },
            "sort": [
                {
                    "create_time": {
                        "order": "DESC"
                    }
                }
            ]
        }
        return openSearchCall(body, url)
    } else {
        const url = "/opensearch/_plugins/_ml/memory/conversation/" + conversationId
        return openSearchNoBodyCall(url)
    }
}
export const getConversations = async () => {
    let url;
    if(await is2dot12plus()) {
        url = "/opensearch/_plugins/_ml/memory/"
    } else {
        url = "/opensearch/_plugins/_ml/memory/conversation/"
    }
    return openSearchNoBodyCall(url)
}
export const deleteConversation = async (conversation_id: string) => {
    console.log("Going to delete", conversation_id)
    let url;
    if(await is2dot12plus()) {
        url = "/opensearch/_plugins/_ml/memory/" + conversation_id
    } else {
        url = "/opensearch/_plugins/_ml/memory/conversation/" + conversation_id
        // hack for empty conversation delete:
        const body = {
            input: "",
            prompt_template: "",
            response: ""
        }
        console.log("Adding interaction")
        const addCall = await openSearchCall(body, url)
        await addCall
    }
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
    const url = "/opensearch/" + index_name + `/_search/`;
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