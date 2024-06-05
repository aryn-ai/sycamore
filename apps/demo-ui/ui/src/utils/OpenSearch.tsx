const SOURCES = [
  "type",
  "_id",
  "doc_id",
  "properties",
  "title",
  "text_representation",
  "bbox",
  "shingles",
];
const SEARCH_PIPELINE = "hybrid_rag_pipeline";
const NO_RAG_SEARCH_PIPELINE = "hybrid_pipeline";
export const FEEDBACK_INDEX_NAME = "feedback";

export const hybridSearch = (
  rephrasedQuestion: string,
  filters: any,
  index_name: string,
  model_id: string,
  rerank: boolean = false,
) => {
  const query: any = {
    _source: SOURCES,
    query: {
      hybrid: {
        queries: [
          {
            bool: {
              must: [
                {
                  exists: {
                    field: "text_representation",
                  },
                },
                {
                  match: {
                    text_representation: rephrasedQuestion,
                  },
                },
              ],
              filter: [
                {
                  match_all: {},
                },
              ],
            },
          },
          {
            neural: {
              embedding: {
                query_text: rephrasedQuestion,
                k: 100,
                model_id: model_id,
                filter: {
                  match_all: {},
                },
              },
            },
          },
        ],
      },
    },
    size: 20,
  };
  if (rerank) {
    query.ext = {
      rerank: {
        query_context: {
          query_text: rephrasedQuestion,
        },
      },
    };
  }
  if (filters != null) {
    // console.log("OS setting filters");
    if (
      query.query.hybrid.queries &&
      query.query.hybrid.queries.length > 0 &&
      query.query.hybrid.queries[0].bool
    ) {
      query.query.hybrid.queries[0].bool.filter = filters["keyword"];
    }
    if (
      query.query.hybrid.queries &&
      query.query.hybrid.queries.length > 0 &&
      query.query.hybrid.queries[1].neural
    ) {
      query.query.hybrid.queries[1].neural.embedding.filter = filters["neural"];
    }
  }
  return query;
};

export const getHybridConversationSearchQuery = (
  question: string,
  rephrasedQuestion: string,
  filters: any,
  index_name: string,
  embeddingModel: string,
  llmModel: string,
  numDocs: number = 7,
  rerank: boolean = false,
) => {
  const query: any = hybridSearch(
    rephrasedQuestion,
    filters,
    index_name,
    embeddingModel,
    rerank,
  );
  query.ext = {
    generative_qa_parameters: {
      llm_question: question,
      context_size: numDocs,
      llm_model: llmModel,
    },
  };
  const url =
    "/opensearch/" + index_name + "/_search?search_pipeline=" + SEARCH_PIPELINE;
  return { query, url };
};

export const hybridSearchNoRag = async (
  rephrasedQuestion: string,
  filters: any,
  index_name: string,
  embeddingModel: string,
  rerank: boolean,
) => {
  const query: any = hybridSearch(
    rephrasedQuestion,
    filters,
    index_name,
    embeddingModel,
    rerank,
  );
  const url =
    "/opensearch/" +
    index_name +
    "/_search?search_pipeline=" +
    NO_RAG_SEARCH_PIPELINE;
  return [openSearchCall(query, url), query];
};

export const hybridConversationSearch = async (
  question: string,
  rephrasedQuestion: string,
  filters: any,
  conversationId: string,
  index_name: string,
  embeddingModel: string,
  llmModel: string,
  numDocs: number = 7,
) => {
  const { query, url } = getHybridConversationSearchQuery(
    question,
    rephrasedQuestion,
    filters,
    index_name,
    embeddingModel,
    llmModel,
    numDocs,
  );

  query.ext.generative_qa_parameters.memory_id = conversationId;

  return [openSearchCall(query, url), query];
};

export const updateInteractionAnswer = async (
  interactionId: any,
  answer: string,
  query: any,
) => {
  console.log("Updating interaction with new answer", interactionId);
  const url = "/opensearch/_plugins/_ml/memory/message/" + interactionId;
  const data = {
    response: answer,
    additional_info: {
      queryUsed: query,
    },
  };
  return openSearchCall(data, url, "PUT");
};

export const getIndices = async () => {
  const url = "/opensearch/_aliases?pretty";
  return openSearchNoBodyCall(url);
};
export const getEmbeddingModels = async () => {
  const body = {
    query: {
      bool: {
        must_not: {
          exists: {
            field: "chunk_number",
          },
        },
        minimum_should_match: 1,
        should: [
          { term: { algorithm: "TEXT_EMBEDDING" } },
          { match_phrase: { description: "embedding" } },
        ],
        must: [{ term: { model_state: "DEPLOYED" } }],
      },
    },
  };
  const url = "/opensearch/_plugins/_ml/models/_search";
  return openSearchCall(body, url);
};
export const createConversation = async (conversationId: string) => {
  const body = {
    name: conversationId,
  };
  let url;
  url = "/opensearch/_plugins/_ml/memory/";
  return openSearchCall(body, url);
};
export const getInteractions = async (conversationId: any) => {
  const url = "/opensearch/_plugins/_ml/memory/" + conversationId + "/_search";
  const body = {
    query: {
      match_all: {},
    },
    sort: [
      {
        create_time: {
          order: "DESC",
        },
      },
    ],
  };
  return openSearchCall(body, url);
};
export const getConversations = async () => {
  const url = "/opensearch/_plugins/_ml/memory/";
  return openSearchNoBodyCall(url);
};
export const deleteConversation = async (conversation_id: string) => {
  console.log("Going to delete", conversation_id);
  const url = "/opensearch/_plugins/_ml/memory/" + conversation_id;
  console.log("Now deleting");
  return openSearchNoBodyCall(url, "DELETE");
};
export const openSearchNoBodyCall = async (
  url: string,
  http_method: string = "GET",
) => {
  try {
    console.log("sending ", http_method + " " + url);
    const response = await fetch(url, {
      method: http_method,
      headers: {
        "Content-Type": "application/json",
      },
    });
    if (!response.ok) {
      return response.text().then((text) => {
        throw new Error(
          `OpenSearchRequest rejected with status ${response.status} and message ${text}`,
        );
      });
    }

    const data = await response.json();
    console.log("Response data:", data);
    return data;
  } catch (error: any) {
    console.error("Error sending query:", error);
    throw new Error(
      "Error making OpenSearch query to " +
        url +
        " without body: " +
        error.message,
    );
  }
};
export const openSearchCall = async (
  query: any,
  url: string,
  http_method: string = "POST",
) => {
  try {
    console.log("sending request", url, JSON.stringify(query));
    const response = await fetch(url, {
      method: http_method,
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(query),
    });
    console.log("inside openSearchCall", JSON.stringify(query, null, 2));

    if (!response.ok) {
      return response.text().then((text) => {
        console.error(
          `Request to ${url}:\n` +
            JSON.stringify(query) +
            `\nrejected with status ${response.status} and message ${text}`,
        );
        throw new Error(`Request to ${url}:\n rejected with message ${text}`);
      });
    }

    const data = await response.json();
    console.log("Response data:", data);
    return data;
  } catch (error: any) {
    console.error("Error sending query:", error);
    throw new Error(
      "Error making OpenSearch to " +
        url +
        " query with body: " +
        error.message,
    );
  }
};

// Legacy for local RAG
export const sendQuery = async (query: any, index_name: string) => {
  const url = "/opensearch/" + index_name + `/_search/`;
  return openSearchCall(query, url);
};

export async function queryOpenSearch(
  question: string,
  index_name: string,
  model_id: string,
) {
  const query = {
    query: {
      bool: {
        should: [
          {
            function_score: {
              query: {
                match: {
                  text_representation: question,
                },
              },
              weight: 0.925,
            },
          },
          {
            neural: {
              embedding: {
                query_text: question,
                model_id: model_id,
                k: 20,
              },
            },
          },
        ],
      },
    },
    size: 10,
    _source: SOURCES,
  };
  console.log("OS question ", question);
  console.log("OS query ", JSON.stringify(query));
  const response = sendQuery(query, index_name);
  return response;
}

export const createFeedbackIndex = async () => {
  const indexMappings = {
    mappings: {
      properties: {
        interaction_id: {
          type: "keyword",
        },
        thumb: {
          type: "keyword",
        },
        conversation_id: {
          type: "keyword",
        },
        comment: {
          type: "text",
        },
      },
    },
  };
  openSearchCall(indexMappings, "/opensearch/" + FEEDBACK_INDEX_NAME, "PUT");
};

export const updateFeedback = async (
  conversationId: string,
  interactionId: string,
  thumb: boolean | null,
  comment: string | null,
) => {
  console.log(thumb);
  const conversationSnapshot = await getInteractions(conversationId);
  const feedbackDoc: any = {
    doc: {
      interaction_id: interactionId,
      conversation_id: conversationId,
      thumb: thumb === null ? "null" : thumb ? "up" : "down",
      comment: comment === null || comment == "" ? "null" : comment,
      conversation_snapshot: conversationSnapshot,
    },
    doc_as_upsert: true,
  };
  const url =
    "/opensearch/" + FEEDBACK_INDEX_NAME + "/_update/" + interactionId;
  openSearchCall(feedbackDoc, url, "POST");
};

export const addNoConversationFeedback = async (
  systemChat: any,
  thumb: boolean | null,
  comment: string | null,
) => {
  console.log("Feedback for ", systemChat);
  const formattedDate: string = new Date().toISOString();
  const feedbackDoc: any = {
    doc: {
      timestamp: formattedDate,
      chat: systemChat,
      thumb: thumb === null ? "null" : thumb ? "up" : "down",
      comment: comment === null || comment == "" ? "null" : comment,
    },
    doc_as_upsert: true,
  };
  const url = "/opensearch/" + FEEDBACK_INDEX_NAME + "/_doc/" + formattedDate;
  openSearchCall(feedbackDoc, url, "POST");
};

export const getFeedback = async (interactionId: string) => {
  const url = "/opensearch/" + FEEDBACK_INDEX_NAME + "/_doc/" + interactionId;
  return openSearchNoBodyCall(url);
};
