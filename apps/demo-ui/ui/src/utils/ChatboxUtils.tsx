import { Dispatch, SetStateAction } from "react";
import { SearchResultDocument } from "../Types";

export function parseFilters(
  filterInputs: any,
  setErrorMessage: Dispatch<SetStateAction<string | null>>,
) {
  if (filterInputs == null) return null;
  const resultNeural: any = {
    bool: {
      filter: [],
    },
  };
  const resultKeyword: any = {
    bool: {
      filter: [],
    },
  };
  Object.entries(filterInputs).forEach(([filter, filterValue]) => {
    if (filter == null || filter == "") return;
    // ignore ntsb schema, handled separately below for auto filters
    if (
      filter == "location" ||
      filter == "airplane_name" ||
      filter == "date_start" ||
      filter == "date_end" ||
      filterValue == "unknown"
    ) {
      return;
    }
    resultNeural["bool"]["filter"].push({
      match: {
        [`properties.${filter}`]: filterValue,
      },
    });
    resultKeyword["bool"]["filter"].push({
      match: {
        [`properties.${filter}.keyword`]: filterValue,
      },
    });
  });

  // for ntsb schema only
  if (
    filterInputs["location"] != null &&
    filterInputs["location"] != "unknown"
  ) {
    resultKeyword["bool"]["filter"].push({
      match: {
        "properties.entity.location": filterInputs["location"],
      },
    });
    resultNeural["bool"]["filter"].push({
      match: {
        "properties.entity.location": filterInputs["location"],
      },
    });
  }
  if (
    filterInputs["airplane_name"] != null &&
    filterInputs["airplane_name"] !== "unknown"
  ) {
    resultKeyword["bool"]["filter"].push({
      match: {
        "properties.entity.aircraft": filterInputs["airplane_name"],
      },
    });
    resultNeural["bool"]["filter"].push({
      match: {
        "properties.entity.aircraft": filterInputs["airplane_name"],
      },
    });
  }

  const range_query: any = {
    range: {
      "properties.entity.day": {},
    },
  };
  if (
    filterInputs["date_start"] != null &&
    filterInputs["date_start"] !== "unknown"
  ) {
    range_query.range["properties.entity.day"].gte = filterInputs["date_start"];
  }
  if (
    filterInputs["date_end"] != null &&
    filterInputs["date_end"] !== "unknown"
  ) {
    range_query.range["properties.entity.day"].lte = filterInputs["date_end"];
  }
  if (
    range_query.range["properties.entity.day"].gte !== undefined ||
    range_query.range["properties.entity.day"].lte !== undefined
  ) {
    resultNeural.bool.filter.push(range_query);
    const keywordRange = {
      range: {
        "properties.entity.day.keyword": {},
      },
    };
    keywordRange.range["properties.entity.day.keyword"] =
      range_query["range"]["properties.entity.day"];
    resultKeyword.bool.filter.push(keywordRange);
  }
  const result = {
    keyword: resultKeyword,
    neural: resultNeural,
  };
  return result;
}

export function parseOpenSearchResults(
  openSearchResponse: any,
  setErrorMessage: Dispatch<SetStateAction<string | null>>,
) {
  if (
    openSearchResponse.error !== undefined &&
    openSearchResponse.error.type === "timeout_exception"
  ) {
    const documents = new Array<SearchResultDocument>();
    const chatResponse = "Timeout from OpenAI";
    const interactionId = "";
    setErrorMessage(chatResponse);
    return {
      documents: documents,
      chatResponse: chatResponse,
      interactionId: interactionId,
    };
  }
  const documents = openSearchResponse.hits.hits.map(
    (result: any, idx: number) => {
      const doc = result._source;
      return new SearchResultDocument({
        id: result._id,
        index: idx + 1,
        title: doc.properties.title ?? "Untitled",
        description: doc.text_representation,
        url: doc.properties._location ?? doc.properties.path,
        relevanceScore: "" + result._score,
        properties: doc.properties,
        bbox: doc.bbox,
      });
    },
  );
  const chatResponse =
    openSearchResponse.ext.retrieval_augmented_generation.answer;
  const interactionId =
    openSearchResponse.ext.retrieval_augmented_generation.interaction_id;
  return {
    documents: documents,
    chatResponse: chatResponse,
    interactionId: interactionId,
  };
}

export function parseOpenSearchResultsOg(openSearchResponse: any) {
  const documents = openSearchResponse.hits.hits.map(
    (result: any, idx: number) => {
      const doc = result._source;
      return new SearchResultDocument({
        id: result._id,
        index: idx + 1,
        title: doc.properties.title ?? "Untitled",
        description: doc.text_representation,
        url: doc.properties._location ?? doc.properties.path,
        relevanceScore: "" + result._score,
        properties: doc.properties,
        bbox: doc.bbox,
      });
    },
  );
  return documents;
}

export const simplifyAnswer = async (question: string, answer: string) => {
  try {
    const response = await fetch("/aryn/simplify", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: question,
        answer: answer,
      }),
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }
    console.log("Simplify response is:", response);
    return response.text();
  } catch (error) {
    console.error("Error simplifying through proxy:", error);
    throw error;
  }
};

export const anthropicRag = async (question: string, os_result: any) => {
  try {
    const response = await fetch("/aryn/anthropic_rag", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: question,
        os_result: os_result,
      }),
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }
    console.log("AnthropicRAG response is:", response);
    return response.text();
  } catch (error) {
    console.error("Error in AnthropicRAG through proxy:", error);
    throw error;
  }
};

export const streamingAnthropicRag = async (
  question: string,
  os_result: any,
  newSystemChat: any,
  setStreamingRagResponse: any,
) => {
  try {
    let result = "";
    const response = await fetch("/aryn/anthropic_rag_streaming", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: question,
        os_result: os_result,
      }),
    });
    if (response.body != null) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      while (true) {
        const startTime = new Date(Date.now());
        console.log("Waiting for streaming response...");
        const { done, value } = await reader.read();
        const elapsed = new Date(Date.now()).getTime() - startTime.getTime();
        console.log("Time for chunk: " + elapsed);
        if (done) break;
        result += decoder.decode(value);
        console.log("Received: " + result);
        setStreamingRagResponse(result);
      }
      newSystemChat.response = result;
      setStreamingRagResponse("");
    }

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }
    console.log("Finished AnthropicRAG with result:", result);
    console.log("AnthropicRAG response is:", response);
  } catch (error) {
    console.error("Error in AnthropicRAG through proxy:", error);
    throw error;
  }
};

export const interpretOsResult = async (
  question: string,
  os_result: string,
) => {
  try {
    const response = await fetch("/aryn/interpret_os_result", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: question,
        os_result: os_result,
      }),
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }
    console.log("Simplify response is:", response);
    return response.text();
  } catch (error) {
    console.error("Error interpret_os_result through proxy:", error);
    throw error;
  }
};

export const thumbToBool = (thumbValue: string) => {
  switch (thumbValue) {
    case "null": {
      return null;
    }
    case "up": {
      return true;
    }
    case "down": {
      return false;
    }
    default: {
      console.log("received unexpected feedback thumb value: " + thumbValue);
      return null;
    }
  }
};
