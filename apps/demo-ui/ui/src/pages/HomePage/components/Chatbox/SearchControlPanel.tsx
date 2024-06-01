import { Button, Chip, Group } from "@mantine/core";
import { Settings } from "../../../../Types";

export const SearchControlPanel = ({
  disableFilters,
  setDisableFilters,
  questionRewriting,
  setQuestionRewriting,
  queryPlanner,
  setQueryPlanner,
  chatHistory,
  setChatHistory,
  openSearchQueryEditorOpenedHandlers,
  settings,
}: {
  disableFilters: any;
  setDisableFilters: any;
  questionRewriting: any;
  setQuestionRewriting: any;
  queryPlanner: boolean;
  setQueryPlanner: any;
  chatHistory: any;
  setChatHistory: any;
  openSearchQueryEditorOpenedHandlers: any;
  settings: Settings;
}) => {
  return (
    <Group position="right" w="100%">
      {settings.required_filters.length > 0 ? (
        <Chip
          size="xs"
          checked={!disableFilters}
          onChange={() => setDisableFilters((v: any) => !v)}
          variant="light"
        >
          Filters
        </Chip>
      ) : null}
      {settings.auto_filter ? (
        <Chip
          key="queryPlanner"
          size="xs"
          checked={queryPlanner}
          onChange={() => setQueryPlanner(!queryPlanner)}
          variant="light"
        >
          Auto-filters
        </Chip>
      ) : null}
      <Button
        compact
        onClick={() => openSearchQueryEditorOpenedHandlers.open()}
        size="xs"
        fz="xs"
        variant="gradient"
        gradient={{ from: "blue", to: "indigo", deg: 90 }}
      >
        Run OpenSearch Query
      </Button>
    </Group>
  );
};
