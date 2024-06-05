import { Dispatch, SetStateAction } from "react";
import { SearchResultDocument } from "../Types";

export function parseManualFilters(filterInputs: any) {
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
  if (filterInputs.location) {
    const matchFilter = {
      match: { "properties.entity.location": filterInputs.location },
    };
    resultNeural.bool.filter.push(matchFilter);
    resultKeyword.bool.filter.push(matchFilter);
  }
  if (filterInputs.aircraftType) {
    const matchFilter = {
      match: {
        "properties.entity.aircraftType": filterInputs.aircraftType,
      },
    };
    resultNeural.bool.filter.push(matchFilter);
    resultKeyword.bool.filter.push(matchFilter);
  }
  if (filterInputs.day_end || filterInputs.day_start) {
    const rangeFilterNeural: any = {
      range: { "properties.entity.day": {} },
    };
    const rangeFilterKeyword: any = {
      range: { "properties.entity.day.keyword": {} },
    };

    if (filterInputs.day_end) {
      rangeFilterNeural.range["properties.entity.day"]["lte"] =
        filterInputs.day_end;
      rangeFilterKeyword.range["properties.entity.day.keyword"]["lte"] =
        filterInputs.day_end;
    }
    if (filterInputs.day_start) {
      rangeFilterNeural.range["properties.entity.day"]["gte"] =
        filterInputs.day_start;
      rangeFilterKeyword.range["properties.entity.day.keyword"]["gte"] =
        filterInputs.day_start;
    }

    resultNeural.bool.filter.push(rangeFilterNeural);
    resultKeyword.bool.filter.push(rangeFilterKeyword);
  }
  const result = {
    keyword: resultKeyword,
    neural: resultNeural,
  };
  return result;
}

export function parseFilters(
  filterInputs: any,
  setErrorMessage: Dispatch<SetStateAction<string | null>>,
) {
  // console.log("filterInputs", filterInputs);

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

  if (filterInputs.matchFilters == null) return null;
  filterInputs.matchFilters.forEach((filter: any) => {
    if (filter == null || filter.fieldName === "") return;
    const matchFilter = {
      match: {
        [filter.fieldName]: filter.fieldValue,
      },
    };
    resultNeural["bool"]["filter"].push(matchFilter);
    resultKeyword["bool"]["filter"].push(matchFilter);
  });

  filterInputs.rangeFilters.forEach((filter: any) => {
    if (filter == null || filter.fieldName === "") return;
    const rangeFilterNeural = {
      range: {
        [filter.fieldName]: {
          gte: filter.gte,
          lte: filter.lte,
        },
      },
    };

    const rangeFilterKeyword = {
      range: {
        [`${filter.fieldName}.keyword`]: {
          gte: filter.gte,
          lte: filter.lte,
        },
      },
    };
    resultNeural.bool.filter.push(rangeFilterNeural);
    resultKeyword.bool.filter.push(rangeFilterKeyword);
  });
  const result = {
    keyword: resultKeyword,
    neural: resultNeural,
  };
  return result;
}

export function parseFiltersForDisplay(filterInputs: any) {
  console.log("inside parseFiltersForDisplay", filterInputs);

  const parsedFilter: any = {};
  filterInputs.matchFilters.forEach((filter: any) => {
    const fieldName = filter.fieldName.split(".").pop();
    if (fieldName === "") return;
    parsedFilter[fieldName] = filter.fieldValue;
  });
  filterInputs.rangeFilters.forEach((filter: any) => {
    const fieldName = filter.fieldName.split(".").pop();
    if (fieldName === "") return;
    parsedFilter[`${fieldName}_start`] = filter.gte;
    parsedFilter[`${fieldName}_end`] = filter.lte;
  });
  return parsedFilter;
}

export function parseAggregationsForDisplay(filterInputs: any) {
  console.log("inside parseAggregationsForDisplay", filterInputs);

  const parsedAggs: any = {};
  filterInputs.termsAggregations.forEach((aggs: any) => {
    const fieldName = aggs.fieldName.split(".").slice(-2).reverse().pop();
    if (fieldName === "") return;
    parsedAggs["terms"] = fieldName;
  });
  filterInputs.cardinalityAggregations.forEach((aggs: any) => {
    const fieldName = aggs.fieldName.split(".").slice(-2).reverse().pop();
    if (fieldName === "") return;
    parsedAggs["cardinality"] = fieldName;
  });
  return parsedAggs;
}

const camelToSnakeCase = (str: string) =>
  str.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`);

export function parseFiltersForRawQuery(filterInputs: any) {
  const openSearchQuery: any = { size: 0 };
  openSearchQuery.query = {
    bool: {
      must: [{ match_all: {} }],
      filter: [],
    },
  };
  openSearchQuery.aggs = {};
  filterInputs.matchFilters.forEach((filter: any) => {
    openSearchQuery.query.bool.filter.push({
      match_phrase: { [filter.fieldName]: filter.fieldValue },
    });
  });
  filterInputs.rangeFilters.forEach((filter: any) => {
    openSearchQuery.query.bool.filter.push({
      range: { [filter.fieldName]: { gte: filter.gte, lte: filter.lte } },
    });
  });
  filterInputs.cardinalityAggregations.forEach((agg: any) => {
    let aggsKeyword =
      camelToSnakeCase(agg.fieldName.split(".").slice(-2).reverse().pop()) +
      "s";
    openSearchQuery.aggs[`unique_${aggsKeyword}`] = {
      cardinality: { field: agg.fieldName },
    };
  });
  filterInputs.termsAggregations.forEach((agg: any) => {
    let aggsKeyword =
      camelToSnakeCase(agg.fieldName.split(".").slice(-2).reverse().pop()) +
      "s";
    openSearchQuery.aggs[`unique_${aggsKeyword}`] = {
      terms: { field: agg.fieldName, size: 1000 },
    };
  });
  return openSearchQuery;
}

export function buildOpenSearchQuery(
  filters: { [key: string]: string },
  aggregations: { [key: string]: string },
) {
  const openSearchQuery: any = { size: 0 };
  openSearchQuery.query = {
    bool: {
      must: [{ match_all: {} }],
      filter: [],
    },
  };
  openSearchQuery.aggs = {};

  if (filters) {
    Object.keys(filters).forEach((key) => {
      const value = filters[key];
      if (!value) return;

      const fieldName = `properties.entity.${key.replace(/_before|_after$/, "")}`;

      if (key === "day_before" || key === "day_after") {
        const rangeType = key === "day_before" ? "lte" : "gte";
        openSearchQuery.query.bool.filter.push({
          range: { [fieldName]: { [rangeType]: value } },
        });
      } else {
        openSearchQuery.query.bool.filter.push({
          match_phrase: { [fieldName]: value },
        });
      }
    });
  }

  if (aggregations) {
    Object.entries(aggregations).forEach(([aggType, fieldName]) => {
      const openSearchFieldName = `properties.entity.${fieldName}.keyword`;
      const aggregationName = `unique_${camelToSnakeCase(fieldName)}s`;

      if (aggType === "cardinality") {
        openSearchQuery.aggs[aggregationName] = {
          [aggType]: { field: openSearchFieldName },
        };
      } else {
        openSearchQuery.aggs[aggregationName] = {
          [aggType]: { field: openSearchFieldName, size: 1000 },
        };
      }
    });
  }

  return openSearchQuery;
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
