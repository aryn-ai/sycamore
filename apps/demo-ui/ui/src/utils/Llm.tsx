import { SearchResultDocument } from "../Types";
import { queryPlannerPrompt } from "./queryPlannerPrompt";

export const rephrase_question_prompt = (text: string, conversation: any[]) => {
  // const sys = "You are a helpful assistant that takes the conversation into context and rephrases sentences by request correcting grammar and spelling mistakes. This question will be fed into a query search engine. \n"
  // const prompt = "Rephrase this question: \n" + text
  const sys =
    "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\
     If there is no conversation, return the question unchanged. Only correct grammar and resolve references, do not change anything else. \n";
  const prompt = "Follow Up Input: " + text + "\nStandalone question: ";
  const openAiPrompt = [
    { role: "system", content: sys },
    ...conversation,
    { role: "user", content: prompt },
  ];
  return openAiPrompt;
};
export const rephraseQuestion = async (
  question: string,
  conversation: any[],
  modelName: string,
) => {
  const prompt = rephrase_question_prompt(question, conversation);
  const chatJson = JSON.stringify({
    stream: false,
    model: modelName,
    messages: prompt,
    max_tokens: question.length,
    temperature: 0.0,
  });
  console.log("sending rephrase prompt: ", chatJson);
  try {
    return fetch("/v1/chat/completions", {
      body: chatJson,
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
  } catch (e: any) {
    throw new Error("Error making OpenAI summarize call: " + e.message);
  }
};

export const get_filters_prompt = (text: string) => {
  const sys = queryPlannerPrompt;

  //   const sys =
  //     "Given the following question, extract the value of the following attributes if available. If you cannot confidently extract the value say unknown.\n" +
  //     "1. location\n \
  //     2. airplane_name\n \
  //     3. date_start in yyyy - mm - dd\n \
  //     4. date_end in yyyy - mm - dd\n \
  //     Return response as a json\
  // ";
  const prompt = text;
  const openAiPrompt = [
    { role: "system", content: sys },
    { role: "user", content: prompt },
  ];
  return openAiPrompt;
};
export const getFilters = async (
  question: string,
  modelName: string = "gpt-3.5-turbo-1106",
) => {
  const prompt = get_filters_prompt(question);
  const chatJson = JSON.stringify({
    stream: false,
    model: modelName,
    messages: prompt,
    temperature: 0.0,
    response_format: { type: "json_object" },
  });
  console.log("sending filtering prompt: ", chatJson);
  try {
    return fetch("/v1/chat/completions", {
      body: chatJson,
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
  } catch (e: any) {
    throw new Error("Error making OpenAI filtering call: " + e.message);
  }
};

// Deprecated, for local hybrid search pipeline only
export const qa_prompt = (question: string, docs: SearchResultDocument[]) => {
  const sys =
    "You are a helpful assistant that answers the users question as accurately taking into context an enumerated list of search results. You always cite search results using [${number}] notation.  Do not cite anything that isn't in the search results.  Do not repeat yourself.  SEARCH RESULTS: \n[SEAC_R]";

  let searchResultString = "";
  docs.forEach((doc) => {
    searchResultString +=
      doc.index + " Title: " + doc.title + "\n" + doc.description;
  });
  const openAiPrompt = [
    { role: "system", content: sys.replace("[SEAC_R]", searchResultString) },
    { role: "user", content: question },
  ];
  return openAiPrompt;
};

export const getAnswer = async (
  question: string,
  docs: SearchResultDocument[],
) => {
  const prompt = qa_prompt(question, docs);
  console.log("sending qa prompt: ", prompt);
  const chatJson = JSON.stringify({
    stream: false,
    model: "gpt-4",
    messages: prompt,
    temperature: 0.0,
  });
  try {
    return fetch("/v1/chat/completions", {
      body: chatJson,
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
  } catch (e: any) {
    throw new Error("Error making OpenAI RAG call: " + e.message);
  }
};
