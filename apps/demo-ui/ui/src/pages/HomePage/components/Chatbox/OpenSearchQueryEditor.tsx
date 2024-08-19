import {
  Button,
  Group,
  JsonInput,
  Modal,
  ScrollArea,
  Text,
  TextInput,
} from "@mantine/core";
import { SystemChat } from "../../../../Types";
import { interpretOsResult } from "../../../../utils/ChatboxUtils";
import { openSearchCall } from "../../../../utils/OpenSearch";
import { IconPlayerPlayFilled } from "@tabler/icons-react";

export const OpenSearchQueryEditor = ({
  openSearchQueryEditorOpened,
  openSearchQueryEditorOpenedHandlers,
  currentOsQuery,
  currentOsUrl,
  setCurrentOsQuery,
  setCurrentOsUrl,
  setLoadingMessage,
  chatHistory,
  setChatHistory,
}: {
  openSearchQueryEditorOpened: boolean;
  openSearchQueryEditorOpenedHandlers: any;
  currentOsQuery: string;
  currentOsUrl: string;
  setCurrentOsQuery: any;
  setCurrentOsUrl: any;
  setLoadingMessage: any;
  chatHistory: any;
  setChatHistory: any;
}) => {
  const handleOsSubmit = (e: React.MouseEvent<HTMLButtonElement>) => {
    runJsonQuery(currentOsQuery, currentOsUrl);
  };

  // json query run
  const runJsonQuery = async (
    newOsJsonQuery: string,
    currentOsQueryUrl: string,
  ) => {
    try {
      openSearchQueryEditorOpenedHandlers.close();
      setLoadingMessage("Processing query...");

      const query = JSON.parse(newOsJsonQuery);
      const populateChatFromRawOsQuery = async (openSearchResults: any) => {
        const openSearchResponse = await openSearchResults;
        // send question and OS results to LLM
        const response = await interpretOsResult(
          newOsJsonQuery,
          JSON.stringify(openSearchResponse, null, 4),
        );
        const length = 10;
        const newSystemChat = new SystemChat({
          id: Math.random()
            .toString(36)
            .substring(2, length + 2),
          response: response,
          queryUsed: "User provided OpenSearch query",
          rawQueryUsed: newOsJsonQuery,
          queryUrl: currentOsQueryUrl,
          rawResults: openSearchResponse,
          interaction_id: "Adhoc, not stored in memory",
          editing: false,
          hits: [],
        });
        setChatHistory([...chatHistory, newSystemChat]);
      };
      await Promise.all([
        openSearchCall(query, currentOsQueryUrl).then(
          populateChatFromRawOsQuery,
        ),
      ]);
    } finally {
      setLoadingMessage(null);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setCurrentOsUrl(e.target.value);
  };

  const jsonPlaceholder = {
    query: {
      match_all: {},
    },
    size: 300,
  };
  return (
    <Modal
      opened={openSearchQueryEditorOpened}
      onClose={openSearchQueryEditorOpenedHandlers.close}
      title="OpenSearch Query Editor"
      size="calc(80vw - 3rem)"
    >
      <Text fz="xs" p="sm">
        Note: If you want a RAG answer, make sure the search pipeline is being
        used. Ensure it's configured in the URL
        (search_pipeline=hybrid_rag_pipeline), and also in the query itself
        (ext.generative_qa_parameters parameters)
      </Text>
      <Group position="apart" grow>
        <TextInput
          size="xs"
          value={currentOsUrl}
          onChange={handleInputChange}
          label="OpenSearch url"
          placeholder="e.g. /opensearch/myindex/_search?"
          p="sm"
          withAsterisk
        />
        <Button
          maw="5rem"
          fz="xs"
          size="xs"
          m="md"
          color="teal"
          leftIcon={<IconPlayerPlayFilled size="0.6rem" />}
          onClick={(e) => handleOsSubmit(e)}
        >
          Run
        </Button>
      </Group>
      <ScrollArea>
        <JsonInput
          value={
            typeof currentOsQuery === "object"
              ? JSON.stringify(currentOsQuery, null, 4)
              : currentOsQuery
          }
          onChange={(newValue) => setCurrentOsQuery(newValue)}
          validationError="Invalid JSON"
          placeholder={"e.g.\n" + JSON.stringify(jsonPlaceholder, null, 4)}
          formatOnBlur
          autosize
          minRows={4}
          data-testid="json-input"
        />
      </ScrollArea>
    </Modal>
  );
};
