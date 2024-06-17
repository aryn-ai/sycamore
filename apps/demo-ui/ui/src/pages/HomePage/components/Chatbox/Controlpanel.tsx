import React, {
  Dispatch,
  FormEvent,
  SetStateAction,
  useEffect,
  useState,
} from "react";
import {
  Button,
  Checkbox,
  Divider,
  Group,
  Modal,
  NativeSelect,
  Stack,
  Text,
} from "@mantine/core";
import { Settings } from "../../../../Types";
import {
  getIndices,
  getEmbeddingModels,
  FEEDBACK_INDEX_NAME,
  createFeedbackIndex,
} from "../../../../utils/OpenSearch";

export const ControlPanel = ({
  settings,
  setSettings,
  controlPanelOpened,
  onControlPanelClose,
  openErrorDialog,
}: {
  settings: Settings;
  setSettings: Dispatch<SetStateAction<Settings>>;
  controlPanelOpened: boolean;
  onControlPanelClose: any;
  openErrorDialog: any;
}) => {
  const [availableIndices, setAvailableIndices] = useState(new Array<string>());
  const [availableEmbeddings, setAvailableEmbeddings] = useState(
    new Array<string>(),
  );
  const [formValues, setFormValues] = useState({
    ragPassageCount: settings.ragPassageCount,
    modelName: settings.modelName,
    openSearchIndex: settings.openSearchIndex,
    embeddingModel: settings.embeddingModel,
  });
  const [isDefault, setIsDefault] = useState(false);

  const getIndicesAndEmbeddings = async () => {
    const [getIndicesResponse, getEmbeddingsResponse] = await Promise.all([
      getIndices(),
      getEmbeddingModels(),
    ]);
    const newIndiciesMaybeWFeedback = Object.keys(getIndicesResponse).filter(
      (key) => !key.startsWith("."),
    );
    let newIndicies;
    if (newIndiciesMaybeWFeedback.includes(FEEDBACK_INDEX_NAME)) {
      newIndicies = newIndiciesMaybeWFeedback.filter(
        (name) => name !== FEEDBACK_INDEX_NAME,
      );
    } else {
      newIndicies = newIndiciesMaybeWFeedback;
      createFeedbackIndex();
    }

    const hits = getEmbeddingsResponse.hits.hits;
    const models = [];
    for (const idx in hits) {
      models.push(hits[idx]._id);
    }

    return [newIndicies, models];
  };
  const doit = async () => {
    try {
      const [indexNames, modelIds] = await getIndicesAndEmbeddings();
      setSettings((settings) => ({
        ...settings,
        openSearchIndex: indexNames[0],
        embeddingModel: modelIds[0],
      }));
      setFormValues((prev) => ({
        ...prev,
        openSearchIndex: indexNames[0],
        embeddingModel: modelIds[0],
      }));
      const storedSettings = localStorage.getItem("defaultSettings");
      if (storedSettings) {
        const storedSettingsJSON = JSON.parse(storedSettings);
        setSettings((settings) => ({
          ...settings,
          ragPassageCount: storedSettingsJSON.ragPassageCount,
          modelName: storedSettingsJSON.modelName,
          openSearchIndex: indexNames.includes(
            storedSettingsJSON.openSearchIndex,
          )
            ? storedSettingsJSON.openSearchIndex
            : settings.openSearchIndex,
          embeddingModel: modelIds.includes(storedSettingsJSON.embeddingModel)
            ? storedSettingsJSON.embeddingModel
            : settings.embeddingModel,
        }));
        setFormValues((prev) => ({
          ...prev,
          ragPassageCount: storedSettingsJSON.ragPassageCount,
          modelName: storedSettingsJSON.modelName,
          openSearchIndex: indexNames.includes(
            storedSettingsJSON.openSearchIndex,
          )
            ? storedSettingsJSON.openSearchIndex
            : prev.openSearchIndex,
          embeddingModel: modelIds.includes(storedSettingsJSON.embeddingModel)
            ? storedSettingsJSON.embeddingModel
            : prev.embeddingModel,
        }));
      }

      setAvailableIndices(indexNames);
      setAvailableEmbeddings(modelIds);
    } catch (error: any) {
      openErrorDialog("Error loading settings: " + error.message);
      console.error("Error loading settings:", error);
    }
  };
  useEffect(() => {
    doit();
  }, []);

  useEffect(() => {
    const reloadIndicesAndEmbeddings = async () => {
      try {
        const [indexNames, modelIds] = await getIndicesAndEmbeddings();
        setAvailableIndices(indexNames);
        setAvailableEmbeddings(modelIds);
      } catch (error: any) {
        openErrorDialog("Error loading settings: " + error.message);
        console.error("Error loading settings:", error);
      }
    };
    if (controlPanelOpened) {
      reloadIndicesAndEmbeddings();
    }
  }, [controlPanelOpened]);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (isDefault) {
      localStorage.setItem("defaultSettings", JSON.stringify(formValues));
    }
    setIsDefault(false);
    setSettings((prevSettings) => ({ ...prevSettings, ...formValues }));
    onControlPanelClose();
  };

  return (
    <Modal
      opened={controlPanelOpened}
      onClose={onControlPanelClose}
      title="Options"
      size="auto"
      centered
    >
      <form onSubmit={handleSubmit}>
        <Stack>
          <Divider orientation="horizontal" />
          <Group position="apart">
            <Text fz="xs">RAG passage count:</Text>
            <NativeSelect
              data={["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]}
              value={formValues.ragPassageCount}
              onChange={(event) => {
                const newCount = +event.currentTarget.value;
                setFormValues((prev) => ({
                  ...prev,
                  ragPassageCount: newCount,
                }));
              }}
            />
          </Group>
          <Group position="apart">
            <Text fz="xs">AI model:</Text>
            <NativeSelect
              data={settings.availableModels}
              value={formValues.modelName}
              onChange={(event) => {
                const newModal = event.currentTarget.value;
                setFormValues((prev) => ({
                  ...prev,
                  modelName: newModal,
                }));
              }}
            />
          </Group>
          <Group position="apart">
            <Text fz="xs">OpenSearch index:</Text>
            <NativeSelect
              data={Array.from(availableIndices)}
              value={formValues["openSearchIndex"]}
              onChange={(event) => {
                const newIndex = event.currentTarget.value;
                setFormValues((prev) => ({
                  ...prev,
                  openSearchIndex: newIndex,
                }));
              }}
            />
          </Group>
          <Group position="apart">
            <Text fz="xs">Embedding Model:</Text>
            <NativeSelect
              data={Array.from(availableEmbeddings)}
              value={formValues.embeddingModel}
              onChange={(event) => {
                const newEmbeddingModal = event.currentTarget.value;
                setFormValues((prev) => ({
                  ...prev,
                  embeddingModel: newEmbeddingModal,
                }));
              }}
            />
          </Group>
          <Group position="right">
            <Checkbox
              label="Set as default"
              checked={isDefault}
              onChange={(e) => setIsDefault(e.target.checked)}
            />
            <Button type="submit">Submit</Button>
          </Group>
        </Stack>
      </form>
    </Modal>
  );
};
