import { Button, Chip, Group, Switch, createStyles } from "@mantine/core";
import { Settings } from "../../../../Types";

const useStyles = createStyles((theme) => ({
  track: {
    width: "30px",
  },
}));

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
  queryAnaylzerSwitchRef,
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
  queryAnaylzerSwitchRef: any;
}) => {
  const { classes } = useStyles();

  return (
    <Group position="right">
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
        <Switch
          w="8rem"
          classNames={{ track: classes.track }}
          key="queryPlanner"
          size="xs"
          radius="xl"
          checked={queryPlanner}
          onChange={() => setQueryPlanner(!queryPlanner)}
          variant="light"
          label="Query analyzer"
          ref={queryAnaylzerSwitchRef}
        />
      ) : null}
      {/* <Button
        compact
        onClick={() => openSearchQueryEditorOpenedHandlers.open()}
        size="xs"
        fz="xs"
        variant="gradient"
        gradient={{ from: "blue", to: "indigo", deg: 90 }}
      >
        Run OpenSearch Query
      </Button> */}
    </Group>
  );
};
