import React, { useEffect } from "react";
import { Dispatch, SetStateAction, useRef, useState } from "react";
import {
  ActionIcon,
  Badge,
  Button,
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
  rem,
  useMantineTheme,
} from "@mantine/core";
import {
  IconChevronRight,
  IconClearAll,
  IconSend,
  IconSettings,
  IconWriting,
  IconX,
} from "@tabler/icons-react";
import { getFilters, rephraseQuestion } from "../../../../utils/Llm";
import {
  FilterValues,
  SearchResultDocument,
  Settings,
  SystemChat,
} from "../../../../Types";
import {
  hybridConversationSearch,
  updateInteractionAnswer,
  createConversation,
  hybridSearchNoRag,
  openSearchCall,
} from "../../../../utils/OpenSearch";
import {
  useDisclosure,
  useMediaQuery,
  useResizeObserver,
} from "@mantine/hooks";
import { ControlPanel } from "./Controlpanel";
import {
  anthropicRag,
  buildOpenSearchQueryFromManualFiltersAggs,
  interpretOsResult,
  parseAggregationsForDisplay,
  parseFilters,
  parseFiltersForDisplay,
  buildOpensearchQueryFromLlmResponse,
  parseManualFilters,
  parseOpenSearchResults,
  parseOpenSearchResultsOg,
  simplifyAnswer,
  streamingAnthropicRag,
  excludeFields,
} from "../../../../utils/ChatboxUtils";
import { SystemChatBox } from "./SystemChatbox";
import { SearchControlPanel } from "./SearchControlPanel";
import { FilterInput } from "./FilterInput";
import { OpenSearchQueryEditor } from "./OpenSearchQueryEditor";
import { AddFilterModal } from "./AddFilterModal";
import { AddAggregationModal } from "./AddAggregationModal";

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
    // height: `calc(100vh - 14.5em)`,
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
  const [bottomContainerRef, bottomContainerRect] = useResizeObserver();
  const [manualFilters, setManualFilters] = useState<FilterValues>({});
  const [addFilterModalOpened, addFilterModalHandlers] = useDisclosure(false);
  const [manualAggregations, setManualAggregations] = useState<{
    [key: string]: string;
  }>({});
  const [addAggregationsModalOpened, addAggregationsModalhandlers] =
    useDisclosure(false);
  const [filterFields, setFilterFields] = useState<string[]>([]);

  useEffect(() => {
    setCurrentOsUrl("/opensearch/" + settings.openSearchIndex + "/_search?");
  }, [settings.openSearchIndex]);

  const scrollToBottom = () => {
    scrollAreaRef.current?.scrollTo({
      top: scrollAreaRef.current.scrollHeight,
    });
  };

  useEffect(() => {
    const apiUrl = `https://localhost:9200/${settings.openSearchIndex}/_mapping`;

    const fetchData = async () => {
      try {
        const response = await fetch(apiUrl);

        if (!response.ok) {
          throw new Error("Network response was not ok.");
        }

        const data = await response.json();
        const retrievedFields = Object.keys(
          data[settings.openSearchIndex].mappings.properties.properties
            .properties,
        ).filter((field) => !excludeFields.includes(field));
        setFilterFields(Array.from(new Set(retrievedFields)));
        console.log(
          "Fields from mappings: ",
          Object.keys(
            data[settings.openSearchIndex].mappings.properties.properties
              .properties,
          ),
        );
      } catch (error) {
        setErrorMessage("Error fetching filter fields");
      }
    };

    fetchData();
  }, [settings.openSearchIndex]);

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory, streaming, loadingMessage]);

  useEffect(() => {
    setManualFilters({});
    setManualAggregations({});
  }, [settings.activeConversation]);

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
      let rawOSQueryFlag: boolean = false;
      let osJsonQuery: any = null;
      let aggregations: any = null;
      if (disableFilters) {
        if (Object.keys(manualAggregations).length !== 0) {
          osJsonQuery = buildOpenSearchQueryFromManualFiltersAggs(
            manualFilters,
            manualAggregations,
          );
          aggregations = manualAggregations;
          filterContent = manualFilters;
          rawOSQueryFlag = true;
        } else if (Object.keys(manualFilters).length !== 0) {
          filters = parseManualFilters(manualFilters);
          filterContent = manualFilters;
          console.log("Only manual filters", JSON.stringify(filters, null, 2));
        } else if (queryPlanner) {
          filterResponse = await getFilters(chatInput, settings.modelName);
          console.log("filterResponse", filterResponse);
          if (filterResponse.ok) {
            const filterData = await filterResponse.json();
            console.log("filterData", filterData);
            const autoFilterRawResult = filterData.choices[0].message.content;
            filters = autoFilterRawResult;
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
              let filterContentGenerated = JSON.parse(autoFilterRawResult);
              if (
                filterContentGenerated?.cardinalityAggregations?.length === 0 &&
                filterContentGenerated?.termsAggregations?.length === 0
              ) {
                filters = parseFilters(filterContentGenerated, setErrorMessage);
                filterContent = parseFiltersForDisplay(filterContentGenerated);
              } else {
                osJsonQuery = buildOpensearchQueryFromLlmResponse(
                  filterContentGenerated,
                );
                filterContent = parseFiltersForDisplay(filterContentGenerated);
                aggregations = parseAggregationsForDisplay(
                  filterContentGenerated,
                );
                console.log(
                  "from buildOpensearchQueryFromLlmResponse",
                  JSON.stringify(osJsonQuery, null, 2),
                  "aggregations",
                  aggregations,
                  "filterContent",
                  filterContent,
                );
                rawOSQueryFlag = true;
              }
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
      if (question.length === 0) {
        setLoadingMessage("Running Opensearch Query");
      } else {
        setLoadingMessage(
          'Querying knowledge database with question: "' + question + '"',
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

      if (rawOSQueryFlag) {
        const populateChatFromRawOsQuery = async (openSearchResults: any) => {
          const openSearchResponse = await openSearchResults;
          const response = await interpretOsResult(
            question,
            JSON.stringify(openSearchResponse, null, 4),
          );
          const length = 10;
          const newSystemChat = new SystemChat({
            id: Math.random()
              .toString(36)
              .substring(2, length + 2),
            response: response,
            queryUsed:
              question.length === 0 ? "Opensearch Query Response" : question,
            rawQueryUsed: osJsonQuery,
            queryUrl: currentOsUrl,
            rawResults: openSearchResponse,
            interaction_id: "Adhoc, not stored in memory",
            editing: false,
            hits: [],
            filterContent: filterContent,
            aggregationsUsed: aggregations,
          });
          setChatHistory([...chatHistory, newSystemChat]);
        };
        console.log("OsJsonQuery: ", osJsonQuery);
        const query = osJsonQuery;
        await Promise.all([
          openSearchCall(query, currentOsUrl).then(populateChatFromRawOsQuery),
        ]);
      } else if (!anthropicRagFlag) {
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
    if (Object.keys(manualAggregations).length > 0) {
      return handleSubmitParallelDocLoad(e);
    }
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

  const handleRemoveFilter = (filterKeyToRemove: string) => {
    setManualFilters((prevFilters) => {
      const updatedFilters = { ...prevFilters };
      delete updatedFilters[filterKeyToRemove];
      return updatedFilters;
    });
  };

  const handleRemoveAgg = (AggKeyToRemove: string) => {
    setManualAggregations((prevAggs) => {
      const updatedAggs = { ...prevAggs };
      delete updatedAggs[AggKeyToRemove];
      return updatedAggs;
    });
  };

  React.useEffect(() => {
    chatInputRef.current?.focus();
  }, [streaming]);

  const FilterBadge = ({
    filterKey,
    filterValue,
  }: {
    filterKey: string;
    filterValue: any;
  }) => {
    return (
      <Badge
        pl="0.5rem"
        pr={0}
        size="md"
        radius="sm"
        rightSection={
          <ActionIcon
            size="xs"
            color="blue"
            radius="xl"
            variant="transparent"
            onClick={() => {
              handleRemoveFilter(filterKey);
            }}
          >
            <IconX size={rem(10)} />
          </ActionIcon>
        }
      >
        {filterKey}: {filterValue}
      </Badge>
    );
  };

  const AggregationBadge = ({
    aggregationType,
    aggregationValue,
  }: {
    aggregationType: string;
    aggregationValue: string;
  }) => {
    return (
      <Badge
        pl="0.5rem"
        pr={0}
        size="md"
        radius="sm"
        rightSection={
          <ActionIcon
            size="xs"
            color="blue"
            radius="xl"
            variant="transparent"
            onClick={() => {
              handleRemoveAgg(aggregationType);
            }}
          >
            <IconX size={rem(10)} />
          </ActionIcon>
        }
      >
        {aggregationType}: {aggregationValue}
      </Badge>
    );
  };

  return (
    <>
      <AddFilterModal
        addFilterModalOpened={addFilterModalOpened}
        addFilterModalHandlers={addFilterModalHandlers}
        filterContent={manualFilters}
        setFilterContent={setManualFilters}
        filterFields={filterFields}
      />
      <AddAggregationModal
        addAggregationsModalOpened={addAggregationsModalOpened}
        addAggregationsModalhandlers={addAggregationsModalhandlers}
        aggregations={manualAggregations}
        setAggregations={setManualAggregations}
        filterFields={filterFields}
      />
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
            style={{
              height: `calc(100vh - ${bottomContainerRect.height}px - 7rem)`,
            }}
            spacing="xs"
          >
            <Image width="4em" src="./logo_only.png" />
            <Text>How can I help you today?</Text>
          </Stack>
        ) : (
          <ScrollArea
            className={classes.chatHistoryContainer}
            viewportRef={scrollAreaRef}
            style={{
              height: `calc(100vh - ${bottomContainerRect.height}px - 7rem)`,
            }}
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
                      setManualFilters={setManualFilters}
                      setManualAggregations={setManualAggregations}
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
      <Container
        ref={bottomContainerRef}
        className={classes.fixedBottomContainer}
      >
        <Group
          position={
            // !disableFilters && settings.required_filters.length > 0
            //   ? "apart"
            //   : "left"
            "apart"
          }
          ml="auto"
          mr="auto"
          p="xs"
          w={mobileScreen ? "90vw" : "70vw"}
          noWrap
        >
          <Stack spacing="xs">
            <Group spacing="xs">
              <Text size="xs">Filters :</Text>
              {Object.entries(manualFilters).length !== 0 && (
                <Tooltip label="Clear All">
                  <ActionIcon onClick={() => setManualFilters({})} size="xs">
                    <IconClearAll stroke={2} />
                  </ActionIcon>
                </Tooltip>
              )}
              {Object.entries(manualFilters).map(
                ([key, value]) => {
                  if (typeof value === "object") {
                    const start = value.gte ?? "";
                    const end = value.lte ?? "";
                    const rangeText =
                      start && end
                        ? `${start} - ${end}`
                        : start
                          ? `> ${start}`
                          : `< ${end}`;
                    return (
                      <FilterBadge
                        key={key}
                        filterKey={key}
                        filterValue={rangeText}
                      />
                    );
                  } else {
                    return (
                      <FilterBadge
                        key={key}
                        filterKey={key}
                        filterValue={value}
                      />
                    );
                  }
                },
                //
              )}

              <Button
                compact
                size="xs"
                fz="xs"
                onClick={() => {
                  addFilterModalHandlers.open();
                }}
                variant="gradient"
                gradient={{ from: "blue", to: "indigo", deg: 90 }}
              >
                + Add
              </Button>
            </Group>
            <Group>
              <Text size="xs">Aggregations :</Text>
              {Object.entries(manualAggregations).map(([key, value]) => (
                <AggregationBadge
                  key={key}
                  aggregationType={key}
                  aggregationValue={value}
                />
              ))}
              <Button
                compact
                size="xs"
                fz="xs"
                onClick={() => {
                  addAggregationsModalhandlers.open();
                }}
                variant="gradient"
                gradient={{ from: "blue", to: "indigo", deg: 90 }}
                disabled={Object.entries(manualAggregations).length !== 0}
              >
                + Add
              </Button>
            </Group>
          </Stack>
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

          {/* {!disableFilters && settings.required_filters.length > 0 ? (
            <FilterInput
              settings={settings}
              filtersInput={filtersInput}
              setFiltersInput={setFiltersInput}
              filterError={filterError}
              setFilterError={setFilterError}
            />
          ) : null} */}
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
            rightSectionWidth="100px"
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
                {(chatInput.length !== 0 ||
                  Object.keys(manualAggregations).length !== 0) && (
                  <ActionIcon
                    size={40}
                    radius="xl"
                    color="#5688b0"
                    sx={(theme) => ({
                      "&:hover": {
                        backgroundColor: theme.colors.gray[2],
                      },
                    })}
                  >
                    <IconChevronRight
                      size="2rem"
                      stroke={1}
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
