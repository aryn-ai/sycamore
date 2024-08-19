import {
  ActionIcon,
  Badge,
  Button,
  Card,
  Collapse,
  Container,
  Flex,
  Group,
  Modal,
  NativeSelect,
  Paper,
  ScrollArea,
  Stack,
  Text,
  TextInput,
  Title,
  UnstyledButton,
  useMantineTheme,
} from "@mantine/core";
import { FilterValues, Settings, SystemChat } from "../../../../Types";
import {
  getHybridConversationSearchQuery,
  hybridConversationSearch,
  hybridSearchNoRag,
  updateInteractionAnswer,
} from "../../../../utils/OpenSearch";
import { useState } from "react";
import { useDisclosure } from "@mantine/hooks";
import {
  IconChevronDown,
  IconChevronRight,
  IconCopy,
  IconEdit,
  IconInfoCircle,
  IconPlayerPlayFilled,
  IconPlus,
  IconX,
} from "@tabler/icons-react";
import { Prism } from "@mantine/prism";
import { DocList } from "./Doclist";
import { Citation } from "./Citation";
import {
  anthropicRag,
  parseFilters,
  parseFiltersForDisplay,
  parseOpenSearchResults,
  simplifyAnswer,
  streamingAnthropicRag,
} from "../../../../utils/ChatboxUtils";
import { FeedbackButtons } from "./FeedbackButtons";

/**
 * This component manages an interaction effectively. It shows the question/answer/hits, and also supports the edit/resubmit functionality.
 * All context here is lost when switching a conversation or refreshing the page.
 */
export const SystemChatBox = ({
  systemChat,
  chatHistory,
  settings,
  handleSubmit,
  setChatHistory,
  setSearchResults,
  setErrorMessage,
  setLoadingMessage,
  setCurrentOsQuery,
  setCurrentOsUrl,
  openSearchQueryEditorOpenedHandlers,
  disableFilters,
  anthropicRagFlag,
  streamingRagResponse,
  setStreamingRagResponse,
  openErrorDialog,
  setManualFilters,
  setManualAggregations,
}: {
  systemChat: SystemChat;
  chatHistory: any;
  settings: Settings;
  handleSubmit: any;
  setChatHistory: any;
  setSearchResults: any;
  setErrorMessage: any;
  setLoadingMessage: any;
  setCurrentOsQuery: any;
  setCurrentOsUrl: any;
  openSearchQueryEditorOpenedHandlers: any;
  disableFilters: boolean;
  anthropicRagFlag: boolean;
  streamingRagResponse: any;
  setStreamingRagResponse: any;
  openErrorDialog: any;
  setManualFilters: any;
  setManualAggregations: any;
}) => {
  const citationRegex = /\[(\d+)\]/g;
  const theme = useMantineTheme();
  const [openSearchResultOpened, openSearchResultHandler] =
    useDisclosure(false);
  const [openSearchQueryModalOpened, openSearchQueryModalHandler] =
    useDisclosure(false);

  const replaceCitationsWithLinks = (text: string) => {
    const cleanedText = text
      .replace(/\[\${(\d+)}\]/g, "[$1]")
      .replace(/\\n/g, "\n"); //handle escaped strings
    const elements: React.ReactNode[] = [];
    let lastIndex = 0;
    if (text == null) return elements;
    cleanedText.replace(
      citationRegex,
      (substring: string, citationNumberRaw: any, index: number) => {
        elements.push(cleanedText.slice(lastIndex, index));
        const citationNumber = parseInt(citationNumberRaw);
        if (citationNumber >= 1 && citationNumber <= systemChat.hits.length) {
          elements.push(
            <Citation
              key={citationNumber}
              document={systemChat.hits[citationNumber - 1]}
              citationNumber={citationNumber}
            />,
          );
        } else {
          elements.push(substring);
        }
        lastIndex = index + substring.length;
        return substring;
      },
    );
    elements.push(cleanedText.slice(lastIndex));
    return elements;
  };

  // for editing
  const { query, url } = getHybridConversationSearchQuery(
    systemChat.queryUsed,
    systemChat.queryUsed,
    parseFilters(systemChat.filterContent ?? {}, setErrorMessage),
    settings.openSearchIndex,
    settings.embeddingModel,
    settings.modelName,
    settings.ragPassageCount,
  );
  const queryUrl = systemChat.queryUrl != "" ? systemChat.queryUrl : url;
  const currentOsQuery =
    systemChat.rawQueryUsed != null && systemChat.rawQueryUsed != ""
      ? systemChat.rawQueryUsed
      : JSON.stringify(query, null, 4);
  const [editing, setEditing] = useState(systemChat.editing);
  const [newQuestion, setNewQuestion] = useState(systemChat.queryUsed);
  const [newFilterContent, setNewFilterContent] = useState(
    systemChat.filterContent ?? {},
  );
  const [newFilterInputDialog, newFilterInputDialoghHandlers] =
    useDisclosure(false);

  const [newFilterType, setNewFilterType] = useState("location");
  const [newFilterValue, setNewFilterValue] = useState("");

  const filters = () => {
    if (systemChat.filterContent == null && !editing) {
      return null;
    }

    const removeFilter = (filterToRemove: any) => {
      console.log("Removing filter: ", filterToRemove);
      const updatedNewFilterContent = { ...newFilterContent };
      delete updatedNewFilterContent[filterToRemove];
      setNewFilterContent(updatedNewFilterContent);
    };

    const editFilter = (filterToEdit: any) => {
      setNewFilterType(filterToEdit);
      setNewFilterValue(newFilterContent[filterToEdit] as string);
      newFilterInputDialoghHandlers.open();
    };
    const FilterBadge = ({
      filterKey,
      filterValue,
    }: {
      filterKey: string;
      filterValue: any;
    }) => {
      return (
        <Badge size="xs" p="xs" radius="sm">
          {filterKey}: {filterValue}
        </Badge>
      );
    };

    const addFilter = () => {
      const handleSubmit = (e: React.FormEvent) => {
        console.log("Adding filter", newFilterType + " " + newFilterValue);
        const updatedNewFilterContent = { ...newFilterContent };
        updatedNewFilterContent[newFilterType] = newFilterValue;
        setNewFilterContent(updatedNewFilterContent);
        newFilterInputDialoghHandlers.close();
      };

      const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setNewFilterValue(e.target.value);
      };

      const handleInputKeyPress = (
        e: React.KeyboardEvent<HTMLInputElement>,
      ) => {
        if (e.key === "Enter") {
          e.preventDefault();
          handleSubmit(e);
        }
      };

      return (
        <Modal
          opened={newFilterInputDialog}
          onClose={newFilterInputDialoghHandlers.close}
          title="Add filter"
          size="auto"
        >
          <Container p="md">
            <NativeSelect
              value={newFilterType}
              label="Field to filter on"
              onChange={(event) => setNewFilterType(event.currentTarget.value)}
              data={[
                { value: "location", label: "Location" },
                { value: "airplane_name", label: "Airplane type" },
                { value: "day_start", label: "Before" },
                { value: "day_end", label: "After" },
              ]}
            />
            <TextInput
              label="Value of filter"
              value={newFilterValue}
              onKeyDown={handleInputKeyPress}
              onChange={handleInputChange}
              autoFocus
              size="sm"
              fz="xs"
              placeholder="e.g. California"
            />
            <Button onClick={(e) => handleSubmit(e)}>Add filter</Button>
          </Container>
        </Modal>
      );
    };

    const editFiltersButtons = (filter: any) => (
      <Group position="right" spacing="0">
        <ActionIcon
          size="0.8rem"
          color="blue"
          radius="md"
          variant="transparent"
          onClick={() => editFilter(filter)}
        >
          <IconEdit size="xs" data-testid="edit-icon-2" />
        </ActionIcon>
        <ActionIcon
          size="0.8rem"
          color="blue"
          radius="md"
          variant="transparent"
          onClick={() => removeFilter(filter)}
        >
          <IconX size="xs" />
        </ActionIcon>
      </Group>
    );

    if (!editing) {
      return (
        <Stack spacing="sm" pb="sm">
          {systemChat.filterContent &&
            Object.keys(systemChat.filterContent).length !== 0 && (
              <Group>
                <Text size="xs">Filters :</Text>

                {Object.entries(systemChat.filterContent as FilterValues).map(
                  ([key, value]) => {
                    if (typeof value === "object") {
                      const start = value.gte ?? "";
                      const end = value.lte ?? "";
                      const rangeText =
                        start && end
                          ? `${start} - ${end}`
                          : start
                            ? `>= ${start}`
                            : `<= ${end}`;
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

                <ActionIcon
                  onClick={() => {
                    setManualFilters(systemChat.filterContent);
                  }}
                  size="xs"
                >
                  <IconCopy stroke={1.5} />
                </ActionIcon>
              </Group>
            )}
          {systemChat.aggregationContent &&
            Object.keys(systemChat.aggregationContent).length !== 0 && (
              <Group>
                <Text size="xs">Aggregations :</Text>
                {Object.keys(systemChat.aggregationContent).map((aggs: any) => {
                  return (
                    <Badge size="xs" key={aggs} p="xs" radius="sm">
                      {aggs}: {systemChat.aggregationContent[aggs]}
                    </Badge>
                  );
                })}

                <ActionIcon
                  onClick={() => {
                    setManualAggregations(systemChat.aggregationContent);
                  }}
                  size="xs"
                >
                  <IconCopy stroke={1.5} />
                </ActionIcon>
              </Group>
            )}
        </Stack>
      );
    } else {
      return (
        <Container mb="sm">
          {addFilter()}
          {Object.keys(newFilterContent).map((filter: any) => {
            return (
              <Badge
                size="xs"
                key={filter}
                p="xs"
                mr="xs"
                rightSection={editFiltersButtons(filter)}
              >
                {filter}: {newFilterContent[filter]}
              </Badge>
            );
          })}
          <UnstyledButton onClick={() => newFilterInputDialoghHandlers.open()}>
            <Badge
              size="xs"
              key="newFilter"
              p="xs"
              mr="xs"
              onClick={() => {
                setEditing(false);
                newFilterInputDialoghHandlers.open();
              }}
            >
              <Group>
                New filter
                <ActionIcon
                  size="0.8rem"
                  color="blue"
                  radius="md"
                  variant="transparent"
                >
                  <IconPlus size="xs" />
                </ActionIcon>
              </Group>
            </Badge>
          </UnstyledButton>
        </Container>
      );
    }
  };
  const handleInputChange = (e: any) => {
    setNewQuestion(e.target.value);
  };

  const handleInputKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      rerunQuery();
    }
  };

  // rerun mechanism
  const rerunQuery = async () => {
    try {
      setEditing(false);
      if (disableFilters) {
        setLoadingMessage("Processing query...");
      } else {
        setLoadingMessage("Processing query with filters...");
      }
      const populateChatFromOs = ({
        openSearchResponse,
        query,
      }: {
        openSearchResponse: any;
        query: any;
      }) => {
        console.log("New filter content is", newFilterContent);
        console.log("Main processor ", openSearchResponse);
        console.log("Main processor: OS results ", openSearchResponse);
        const endTime = new Date(Date.now());
        const elpased = endTime.getTime() - startTime.getTime();
        console.log("Main processor: OS took seconds: ", elpased);
        const parsedOpenSearchResults = parseOpenSearchResults(
          openSearchResponse,
          setErrorMessage,
        );
        const newSystemChat = new SystemChat({
          id: parsedOpenSearchResults.interactionId + "_system",
          interaction_id: parsedOpenSearchResults.interactionId,
          response: parsedOpenSearchResults.chatResponse,
          ragPassageCount: settings.ragPassageCount,
          modelName: settings.modelName,
          rawQueryUsed: JSON.stringify(query, null, 4),
          queryUrl: queryUrl,
          queryUsed: newQuestion,
          hits: parsedOpenSearchResults.documents,
          filterContent: newFilterContent,
        });
        setChatHistory([...chatHistory, newSystemChat]);
      };

      const clean = async (result: any) => {
        const openSearchResponseAsync = result[0];
        const query = result[1];
        const openSearchResponse = await openSearchResponseAsync;
        let generatedAnswer =
          openSearchResponse.ext.retrieval_augmented_generation.answer;
        if (settings.simplify && openSearchResponse.hits.hits.length > 0) {
          console.log("Simplifying answer: ", generatedAnswer);
          generatedAnswer = await simplifyAnswer(newQuestion, generatedAnswer);
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
          generatedAnswer = await simplifyAnswer(newQuestion, generatedAnswer);
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
          generatedAnswer = await anthropicRag(newQuestion, openSearchResponse);
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
        console.log("Anthropic processor: OS results ", openSearchResponse);
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
          queryUsed: newQuestion,
          hits: parsedOpenSearchResults.documents,
          filterContent: newFilterContent,
        });
        setChatHistory([newSystemChat, ...chatHistory]);
        if (openSearchResponse.hits.hits.length > 0) {
          await streamingAnthropicRag(
            newQuestion,
            openSearchResponse,
            newSystemChat,
            setStreamingRagResponse,
          );
        } else {
          newSystemChat.response = "No search results";
        }
      };

      const startTime = new Date(Date.now());
      if (!anthropicRagFlag) {
        await Promise.all([
          hybridConversationSearch(
            newQuestion,
            newQuestion,
            parseFilters(newFilterContent, setErrorMessage),
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
              console.error("Error sending opensearch query:", error);
            }),
        ]);
      } else {
        await Promise.all([
          hybridSearchNoRag(
            newQuestion,
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
    } finally {
      setLoadingMessage(null);
    }
  };

  return (
    <Card
      key={systemChat.id}
      padding="lg"
      radius="md"
      sx={{
        borderStyle: "none none solid none",
        borderColor: "#eee;",
        overflow: "visible",
      }}
    >
      <Modal
        opened={openSearchQueryModalOpened}
        onClose={openSearchQueryModalHandler.close}
        title="Opensearch Query Used"
        size="60vw"
      >
        <Flex>
          <ScrollArea w="100%">
            <Prism language="markdown">
              {JSON.stringify(systemChat.rawQueryUsed, null, 4)}
            </Prism>
          </ScrollArea>
        </Flex>
      </Modal>
      {systemChat.rawQueryUsed && (
        <ActionIcon
          style={{ position: "absolute", right: "1.25rem", top: "1.25rem" }}
          onClick={openSearchQueryModalHandler.open}
        >
          <IconInfoCircle stroke={2} />
        </ActionIcon>
      )}
      <Group spacing="xs">
        {editing ? (
          <Group p="0">
            <ActionIcon size="xs" mr="0">
              <IconX
                onClick={(v) => {
                  setEditing(false);
                  setNewQuestion(systemChat.queryUsed);
                }}
                data-testid="close-icon"
              />
            </ActionIcon>
          </Group>
        ) : (
          <ActionIcon size="xs" mr="0">
            <IconEdit
              onClick={(v) => {
                setEditing(true);
              }}
              data-testid="edit-icon"
            />
          </ActionIcon>
        )}
        {editing ? (
          <TextInput
            variant="unstyled"
            w="90%"
            value={newQuestion}
            size="md"
            onKeyDown={handleInputKeyPress}
            onChange={handleInputChange}
            data-testid="edit-input"
          ></TextInput>
        ) : (
          <Text size="xl" fw={450} p="xs" pl="0" pt="0">
            {systemChat.queryUsed}
          </Text>
        )}
      </Group>

      {settings.auto_filter ? filters() : null}

      {editing ? (
        <Group p="md">
          <Button
            fz="xs"
            size="xs"
            color="teal"
            leftIcon={<IconPlayerPlayFilled size="0.6rem" />}
            onClick={(v) => {
              rerunQuery();
            }}
          >
            Run
          </Button>
          <Button
            variant="light"
            color="teal"
            fz="xs"
            size="xs"
            onClick={() => {
              setCurrentOsQuery(currentOsQuery);
              setCurrentOsUrl(queryUrl);
              openSearchQueryEditorOpenedHandlers.open();
            }}
          >
            OpenSearch query editor
          </Button>
        </Group>
      ) : null}

      <Text
        size="sm"
        sx={{ whiteSpace: "pre-wrap" }}
        color={editing ? theme.colors.gray[4] : "black"}
        p="xs"
      >
        {/* {textNodes} */}
        {replaceCitationsWithLinks(systemChat.response)}
        {systemChat.rawResults != null ? (
          <Paper withBorder mt="md">
            <Group
              onClick={openSearchResultHandler.toggle}
              style={{ cursor: "pointer" }}
            >
              {openSearchResultOpened ? (
                <IconChevronDown color="grey" />
              ) : (
                <IconChevronRight color="grey" />
              )}
              <Text>OpenSearch results</Text>
            </Group>
            <Collapse in={openSearchResultOpened}>
              <Flex>
                <ScrollArea mah="45rem" w="100%">
                  <Prism language="markdown">
                    {JSON.stringify(systemChat.rawResults, null, 4)}
                  </Prism>
                </ScrollArea>
              </Flex>
            </Collapse>
          </Paper>
        ) : null}
      </Text>
      <DocList
        documents={systemChat.hits}
        settings={settings}
        docsLoading={false}
      ></DocList>
      <Stack p="xs" spacing="0">
        {systemChat.originalQuery !== "" &&
        systemChat.originalQuery !== systemChat.queryUsed ? (
          <Text fz="xs" fs="italic" color="dimmed">
            Original Query: {systemChat.originalQuery}
          </Text>
        ) : null}
        <Text fz="xs" fs="italic" color="dimmed">
          Interaction id:{" "}
          {systemChat.interaction_id ? systemChat.interaction_id : ""}
        </Text>
      </Stack>

      <FeedbackButtons systemChat={systemChat} settings={settings} />
    </Card>
  );
};
