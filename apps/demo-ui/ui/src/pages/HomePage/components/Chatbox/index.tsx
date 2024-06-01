import React, { useEffect } from "react";
import { Dispatch, SetStateAction, useRef, useState } from "react";
import {
  ActionIcon,
  Center,
  Container,
  Flex,
  Group,
  Image,
  Loader,
  ScrollArea,
  Skeleton,
  Stack,
  Text,
  TextInput,
  Tooltip,
  createStyles,
  useMantineTheme,
} from "@mantine/core";
import {
  IconChevronRight,
  IconSettings,
  IconWriting,
} from "@tabler/icons-react";
import { getFilters, rephraseQuestion } from "../../../../utils/Llm";
import { SearchResultDocument, Settings, SystemChat } from "../../../../Types";
import {
  hybridConversationSearch,
  updateInteractionAnswer,
  createConversation,
  hybridSearchNoRag,
} from "../../../../utils/OpenSearch";
import { useDisclosure, useMediaQuery } from "@mantine/hooks";
import { ControlPanel } from "./Controlpanel";
import {
  anthropicRag,
  parseFilters,
  parseOpenSearchResults,
  parseOpenSearchResultsOg,
  simplifyAnswer,
  streamingAnthropicRag,
} from "../../../../utils/ChatboxUtils";
import { SystemChatBox } from "./SystemChatbox";
import { SearchControlPanel } from "./SearchControlPanel";
import { FilterInput } from "./FilterInput";
import { OpenSearchQueryEditor } from "./OpenSearchQueryEditor";

const useStyles = createStyles((theme) => ({
  inputBar: {
    width: "50vw",
    [theme.fn.smallerThan("sm")]: {
      width: "100%",
    },
  },
  fixedBottomContainer: {
    margin: 0,
    maxWidth: "none",
    bottom: 0,
    borderTop: "1px solid lightgrey",
    flex: 1,
  },
  settingsIcon: {
    zIndex: 1,
  },
  chatHistoryContainer: {
    height: `calc(100vh - 14.5em)`,
  },
  settingsStack: {
    position: "absolute",
    top: 10,
    right: 0,
    alignItems: "end",
  },
  chatFlex: {
    paddingBottom: 0,
  },
}));

const LoadingChatBox = ({
  loadingMessage,
}: {
  loadingMessage: string | null;
}) => {
  const theme = useMantineTheme();
  return (
    <Container ml={theme.spacing.xl} p="lg" miw="80%">
      {/* <Skeleton height={50} circle mb="xl" /> */}
      <Text size="xs" fs="italic" fw="400" p="xs">
        {loadingMessage ? loadingMessage : null}
      </Text>
      <Skeleton height={8} radius="xl" />
      <Skeleton height={8} mt={6} radius="xl" />
      <Skeleton height={8} mt={6} width="70%" radius="xl" />
    </Container>
  );
};

export const ChatBox = ({
  chatHistory,
  searchResults,
  setChatHistory,
  setSearchResults,
  streaming,
  setStreaming,
  setDocsLoading,
  setErrorMessage,
  settings,
  setSettings,
  refreshConversations,
  chatInputRef,
  openErrorDialog,
}: {
  chatHistory: SystemChat[];
  searchResults: SearchResultDocument[];
  setChatHistory: Dispatch<SetStateAction<any[]>>;
  setSearchResults: Dispatch<SetStateAction<any[]>>;
  streaming: boolean;
  setStreaming: Dispatch<SetStateAction<boolean>>;
  setDocsLoading: Dispatch<SetStateAction<boolean>>;
  setErrorMessage: Dispatch<SetStateAction<string | null>>;
  settings: Settings;
  setSettings: Dispatch<SetStateAction<Settings>>;
  refreshConversations: any;
  chatInputRef: any;
  openErrorDialog: any;
}) => {
  const theme = useMantineTheme();
  const [chatInput, setChatInput] = useState("");
  const [anthropicRagFlag, setAnthropicRagFlag] = useState(false);
  const [disableFilters, setDisableFilters] = useState(settings.auto_filter);
  const [queryPlanner, setQueryPlanner] = useState(settings.auto_filter);
  const [questionRewriting, setQuestionRewriting] = useState(false);
  const [filtersInput, setFiltersInput] = useState<{ [key: string]: string }>(
    {},
  );
  const [loadingMessage, setLoadingMessage] = useState<string | null>(null);
  const [currentOsQuery, setCurrentOsQuery] = useState<string>("");
  const [currentOsUrl, setCurrentOsUrl] = useState<string>(
    "/opensearch/" + settings.openSearchIndex + "/_search?",
  );
  const [openSearchQueryEditorOpened, openSearchQueryEditorOpenedHandlers] =
    useDisclosure(false);
  const { classes } = useStyles();
  const mobileScreen = useMediaQuery(`(max-width: ${theme.breakpoints.sm})`);
  const [settingsOpened, settingsHandler] = useDisclosure(false);
  const [filterError, setFilterError] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const [streamingRagResponse, setStreamingRagResponse] = useState("");

  useEffect(() => {
    setCurrentOsUrl("/opensearch/" + settings.openSearchIndex + "/_search?");
  }, [settings.openSearchIndex]);

  const scrollToBottom = () => {
    scrollAreaRef.current?.scrollTo({
      top: scrollAreaRef.current.scrollHeight,
    });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory, streaming]);

  // This method does all the search workflow execution
  const handleSubmitParallelDocLoad = async (e: React.FormEvent) => {
    try {
      e.preventDefault();
      if (chatInputRef.current != null) {
        chatInputRef.current.disabled = true;
      }

      setStreaming(true);
      setDocsLoading(true);
      console.log("Rephrasing question: ", chatInput);
      // Generate conversation text list
      const chatHistoryInteractions = chatHistory.map((chat) => {
        if ("query" in chat) {
          return { role: "user", content: chat.query };
        } else {
          return { role: "system", content: chat.response ?? "" };
        }
      });
      let filterResponse;
      let filters: any;
      let filterContent: any = null;
      if (!disableFilters) {
        if (queryPlanner) {
          filterResponse = await getFilters(chatInput, settings.modelName);
          console.log(filterResponse);
          if (filterResponse.ok) {
            const filterData = await filterResponse.json();
            const autoFilterRawResult = filterData.choices[0].message.content;
            if (
              autoFilterRawResult.error !== undefined &&
              autoFilterRawResult.error.type === "timeout_exception"
            ) {
              const documents = new Array<SearchResultDocument>();
              const chatResponse = "Timeout from OpenAI";
              const interactionId = "";
              setErrorMessage(chatResponse);
              return null;
            }
            try {
              filterContent = JSON.parse(autoFilterRawResult);
              filters = parseFilters(filterContent, setErrorMessage);
            } catch (error) {
              console.error("Error parsing JSON:", error);
            }
          }
        } else if (settings.required_filters.length > 0) {
          filters = parseFilters(filtersInput, setErrorMessage);
          filterContent = filtersInput;
          if (
            filters["keyword"]["bool"]["filter"].length !=
            settings.required_filters.length
          ) {
            throw new Error("All required filters not populated");
          }
        }
      } else {
        filters = null;
      }
      console.log("Filters are: ", filters);
      let question: string = chatInput;
      const originalQuestion: string = question;
      if (questionRewriting) {
        setLoadingMessage("Rephrasing question with conversation context");
        const rephraseQuestionResponse = await rephraseQuestion(
          chatInput,
          chatHistoryInteractions,
          settings.modelName,
        );
        const responseData = await rephraseQuestionResponse.json();
        const rephrasedQuestion = responseData.choices[0].message.content;
        console.log("Rephrased question to ", rephrasedQuestion);
        question = rephrasedQuestion;
      }
      console.log("Question is: ", question);

      setLoadingMessage(
        'Querying knowledge database with question: "' + question + '"',
      );
      if (filters != null) {
        setLoadingMessage(
          'Using filter: "' +
            JSON.stringify(filters) +
            '". Generating answer..',
        );
      }

      const clean = async (result: any) => {
        const openSearchResponseAsync = result[0];
        const query = result[1];
        const openSearchResponse = await openSearchResponseAsync;
        let generatedAnswer =
          openSearchResponse.ext.retrieval_augmented_generation.answer;
        if (settings.simplify && openSearchResponse.hits.hits.length > 0) {
          console.log("Simplifying answer: ", generatedAnswer);
          setLoadingMessage("Simplifying answer..");
          generatedAnswer = await simplifyAnswer(question, generatedAnswer);
        }
        await updateInteractionAnswer(
          openSearchResponse.ext.retrieval_augmented_generation.interaction_id,
          generatedAnswer,
          query,
        );
        openSearchResponse.ext.retrieval_augmented_generation.answer =
          generatedAnswer;
        return openSearchResponse;
      };

      const clean_rag = async ({
        openSearchResponse,
        query,
      }: {
        openSearchResponse: any;
        query: any;
      }) => {
        let generatedAnswer =
          openSearchResponse.ext.retrieval_augmented_generation.answer;
        if (settings.simplify && openSearchResponse.hits.hits.length > 0) {
          console.log("Simplifying answer: ", generatedAnswer);
          generatedAnswer = await simplifyAnswer(question, generatedAnswer);
        }
        await updateInteractionAnswer(
          openSearchResponse.ext.retrieval_augmented_generation.interaction_id,
          generatedAnswer,
          query,
        );
        openSearchResponse.ext.retrieval_augmented_generation.answer =
          generatedAnswer;
        return { openSearchResponse, query };
      };

      const anthropic_rag_og = async (result: any) => {
        const openSearchResponseAsync = result[0];
        const query = result[1];
        const openSearchResponse = await openSearchResponseAsync;
        let generatedAnswer = "Error";
        if (openSearchResponse.hits.hits.length > 0) {
          console.log("Anthropic RAG time...");
          generatedAnswer = await anthropicRag(question, openSearchResponse);
        }
        openSearchResponse["ext"] = {
          retrieval_augmented_generation: {
            answer: generatedAnswer,
          },
        };
        return { openSearchResponse, query };
      };

      const anthropic_rag = async (result: any) => {
        const openSearchResponseAsync = result[0];
        const openSearchResponse = await openSearchResponseAsync;

        console.log("Anthropic processor ", openSearchResponse);
        const endTime = new Date(Date.now());
        const elpased = endTime.getTime() - startTime.getTime();
        console.log("Anthropic processor: OS took seconds: ", elpased);
        const parsedOpenSearchResults = parseOpenSearchResults(
          openSearchResponse,
          setErrorMessage,
        );
        const newSystemChat = new SystemChat({
          id: parsedOpenSearchResults.interactionId + "_system",
          ragPassageCount: settings.ragPassageCount,
          modelName: settings.modelName,
          queryUsed: question,
          originalQuery: originalQuestion,
          hits: parsedOpenSearchResults.documents,
          filterContent: filterContent,
        });
        setChatHistory([newSystemChat, ...chatHistory]);
        if (openSearchResponse.hits.hits.length > 0) {
          await streamingAnthropicRag(
            question,
            openSearchResponse,
            newSystemChat,
            setStreamingRagResponse,
          );
        } else {
          newSystemChat.response = "No search results";
        }
      };

      const populateChatFromOs = (openSearchResults: any) => {
        console.log("Main processor ", openSearchResults);
        console.log("Main processor: OS results ", openSearchResults);
        const endTime = new Date(Date.now());
        const elpased = endTime.getTime() - startTime.getTime();
        console.log("Main processor: OS took seconds: ", elpased);
        const parsedOpenSearchResults = parseOpenSearchResults(
          openSearchResults,
          setErrorMessage,
        );
        const newSystemChat = new SystemChat({
          id: parsedOpenSearchResults.interactionId + "_system",
          interaction_id: parsedOpenSearchResults.interactionId,
          response: parsedOpenSearchResults.chatResponse,
          ragPassageCount: settings.ragPassageCount,
          modelName: settings.modelName,
          queryUsed: question,
          originalQuery: originalQuestion,
          hits: parsedOpenSearchResults.documents,
          filterContent: filterContent,
        });
        setChatHistory([...chatHistory, newSystemChat]);
      };
      const populateDocsFromOs = (openSearchResults: any) => {
        console.log("Info separate processor ", openSearchResults);
        console.log("Info separate processor : OS results ", openSearchResults);
        const endTime = new Date(Date.now());
        const elpased = endTime.getTime() - startTime.getTime();
        console.log("Info separate processor : OS took seconds: ", elpased);
        const parsedOpenSearchResults =
          parseOpenSearchResultsOg(openSearchResults);
        setSearchResults(parsedOpenSearchResults);
        console.log(
          "Info separate processor : set docs in independent thread to: ",
          parsedOpenSearchResults,
        );
        setDocsLoading(false);
      };
      const startTime = new Date(Date.now());
      if (!settings.activeConversation) {
        const conversationId = await createConversation(chatInput);
        settings.activeConversation = conversationId.memory_id;
        setSettings(settings);
        refreshConversations();
      }

      if (!anthropicRagFlag) {
        await Promise.all([
          hybridConversationSearch(
            chatInput,
            question,
            filters,
            settings.activeConversation,
            settings.openSearchIndex,
            settings.embeddingModel,
            settings.modelName,
            settings.ragPassageCount,
          )
            .then(clean)
            .then(populateChatFromOs)
            .catch((error: any) => {
              openErrorDialog(
                "Error sending opensearch query:" + error.message,
              );
              console.error("Error sending opensearch query:" + error);
            }),
        ]);
      } else {
        await Promise.all([
          hybridSearchNoRag(
            question,
            parseFilters(filters, setErrorMessage),
            settings.openSearchIndex,
            settings.embeddingModel,
            true,
          )
            .then(anthropic_rag_og)
            .then(clean_rag)
            .then(populateChatFromOs)
            .catch((error: any) => {
              openErrorDialog("Error sending query:" + error.message);
              console.error("Error sending query:" + error);
            }),
        ]);
      }
    } catch (e) {
      console.log(e);
      if (typeof e === "string") {
        setErrorMessage(e.toUpperCase());
      } else if (e instanceof Error) {
        setErrorMessage(e.message);
      }
    } finally {
      setStreaming(false);
      setChatInput("");
      setDocsLoading(false);
      setLoadingMessage(null);
      if (chatInputRef.current != null) {
        chatInputRef.current.disabled = false;
      }
    }
  };

  // This method does all the search workflow execution
  const handleSubmit = async (e: React.FormEvent) => {
    if (chatInput.length === 0) {
      return;
    }
    if (!disableFilters && settings.required_filters.length > 0) {
      const someNonEmptyValues =
        Object.keys(filtersInput).length === 0 ||
        Object.keys(filtersInput).some((key) => filtersInput[key] === "");
      if (someNonEmptyValues) {
        setFilterError(true);
        return;
      }
    }
    return handleSubmitParallelDocLoad(e);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setChatInput(e.target.value);
  };

  const handleInputKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  React.useEffect(() => {
    chatInputRef.current?.focus();
  }, [streaming]);

  return (
    <>
      <OpenSearchQueryEditor
        openSearchQueryEditorOpened={openSearchQueryEditorOpened}
        openSearchQueryEditorOpenedHandlers={
          openSearchQueryEditorOpenedHandlers
        }
        currentOsQuery={currentOsQuery}
        currentOsUrl={currentOsUrl}
        setCurrentOsQuery={setCurrentOsQuery}
        setCurrentOsUrl={setCurrentOsUrl}
        setLoadingMessage={setLoadingMessage}
        chatHistory={chatHistory}
        setChatHistory={setChatHistory}
      />
      <ControlPanel
        settings={settings}
        setSettings={setSettings}
        controlPanelOpened={settingsOpened}
        onControlPanelClose={settingsHandler.close}
        openErrorDialog={openErrorDialog}
      />
      <Flex direction="column" pos="relative" className={classes.chatFlex}>
        <Stack className={classes.settingsStack} spacing="0">
          <Tooltip label="Options">
            <ActionIcon
              variant="transparent"
              className={classes.settingsIcon}
              onClick={settingsHandler.open}
            >
              <IconSettings size="1.625rem" />
            </ActionIcon>
          </Tooltip>
        </Stack>
        {chatHistory.length === 0 && !loadingMessage && !streaming ? (
          <Stack
            align="center"
            justify="center"
            className={classes.chatHistoryContainer}
            spacing="xs"
          >
            <Image width="4em" src="./logo_only.png" />
            <Text>How can I help you today?</Text>
          </Stack>
        ) : (
          <ScrollArea
            className={classes.chatHistoryContainer}
            viewportRef={scrollAreaRef}
          >
            <Container>
              <Stack>
                {chatHistory.map((chat, index) => {
                  return (
                    <SystemChatBox
                      key={chat.id + "_system"}
                      systemChat={chat}
                      chatHistory={chatHistory}
                      settings={settings}
                      handleSubmit={handleSubmit}
                      setChatHistory={setChatHistory}
                      setSearchResults={setSearchResults}
                      setErrorMessage={setErrorMessage}
                      setLoadingMessage={setLoadingMessage}
                      setCurrentOsQuery={setCurrentOsQuery}
                      setCurrentOsUrl={setCurrentOsUrl}
                      openSearchQueryEditorOpenedHandlers={
                        openSearchQueryEditorOpenedHandlers
                      }
                      disableFilters={disableFilters}
                      anthropicRagFlag={anthropicRagFlag}
                      streamingRagResponse={streamingRagResponse}
                      setStreamingRagResponse={setStreamingRagResponse}
                      openErrorDialog={openErrorDialog}
                    />
                  );
                })}
                {loadingMessage ? (
                  <LoadingChatBox loadingMessage={loadingMessage} />
                ) : null}

                <Center>
                  {streaming ? <Loader size="xs" variant="dots" m="md" /> : ""}
                </Center>
              </Stack>
            </Container>
          </ScrollArea>
        )}
      </Flex>
      <Container className={classes.fixedBottomContainer}>
        <Group
          position={
            !disableFilters && settings.required_filters.length > 0
              ? "apart"
              : "left"
          }
          ml="auto"
          mr="auto"
          p="sm"
          h="3.5em"
          w={mobileScreen ? "90vw" : "70vw"}
        >
          <SearchControlPanel
            disableFilters={disableFilters}
            setDisableFilters={setDisableFilters}
            questionRewriting={questionRewriting}
            setQuestionRewriting={setQuestionRewriting}
            queryPlanner={queryPlanner}
            setQueryPlanner={setQueryPlanner}
            chatHistory={chatHistory}
            setChatHistory={setChatHistory}
            openSearchQueryEditorOpenedHandlers={
              openSearchQueryEditorOpenedHandlers
            }
            settings={settings}
          ></SearchControlPanel>

          {!disableFilters && settings.required_filters.length > 0 ? (
            <FilterInput
              settings={settings}
              filtersInput={filtersInput}
              setFiltersInput={setFiltersInput}
              filterError={filterError}
              setFilterError={setFilterError}
            />
          ) : null}
        </Group>
        <Center>
          <TextInput
            className={classes.inputBar}
            onKeyDown={handleInputKeyPress}
            onChange={handleInputChange}
            ref={chatInputRef}
            value={chatInput}
            radius="xl"
            autoFocus
            size="lg"
            rightSectionWidth="auto"
            rightSection={
              <Group pr="0.2rem">
                <Tooltip label="Rewrite question">
                  <ActionIcon
                    size={40}
                    radius="xl"
                    sx={(theme) => ({
                      color: questionRewriting ? "white" : "#5688b0",
                      backgroundColor: questionRewriting ? "#5688b0" : "white",
                      "&:hover": {
                        backgroundColor: questionRewriting
                          ? "#5688b0"
                          : theme.colors.gray[2],
                      },
                    })}
                  >
                    <IconWriting
                      size="1.3rem"
                      stroke={1.5}
                      onClick={() => {
                        setQuestionRewriting((o) => !o);
                        chatInputRef?.current?.focus();
                      }}
                    />
                  </ActionIcon>
                </Tooltip>
                {mobileScreen && (
                  <ActionIcon
                    size={40}
                    radius="xl"
                    bg="#5688b0"
                    variant="filled"
                  >
                    <IconChevronRight
                      size="1rem"
                      stroke={2}
                      onClick={handleSubmit}
                    />
                  </ActionIcon>
                )}
              </Group>
            }
            placeholder="Ask me anything"
            disabled={settings.activeConversation == null}
          />
        </Center>
      </Container>
    </>
  );
};
