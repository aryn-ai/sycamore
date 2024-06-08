import {
  Button,
  Group,
  Modal,
  NativeSelect,
  SegmentedControl,
  Select,
  Stack,
  TextInput,
} from "@mantine/core";
import { useEffect, useState } from "react";
import { documentLevelFields } from "../../../../utils/ChatboxUtils";

export const AddAggregationModal = ({
  addAggregationsModalOpened,
  addAggregationsModalhandlers,
  aggregations,
  setAggregations,
  filterFields,
}: {
  addAggregationsModalOpened: boolean;
  addAggregationsModalhandlers: any;
  aggregations: any;
  setAggregations: any;
  filterFields: string[];
}) => {
  const [aggregationType, setAggregationType] = useState("terms");
  const [aggregationValue, setAggregationValue] = useState<string | null>("");
  const [fieldsData, setFieldsData] = useState<
    { value: string; label: string; group: string }[]
  >([]);

  useEffect(() => {
    const retrievedFieldsData = filterFields.map((field) => ({
      value: field,
      label: field,
    }));
    const originalFieldsData = [
      { value: "location", label: "Location" },
      { value: "aircraftType", label: "Aircraft type" },
      { value: "accidentNumber", label: "Accident number" },
    ];
    const AllFields = [...originalFieldsData, ...retrievedFieldsData];
    const groupedFieldsData = AllFields.map((field) => ({
      ...field,
      group: documentLevelFields.includes(field.value)
        ? "Document Level Field"
        : "Entity Specific Field",
    }));

    setFieldsData(groupedFieldsData);
    console.log(retrievedFieldsData);
    console.log();
  }, [filterFields]);

  const handleSubmit = (e: React.FormEvent) => {
    if (aggregationValue && aggregationValue.length !== 0) {
      setAggregations((prevAggs: any) => ({
        [aggregationType]: aggregationValue,
      }));
      setAggregationValue("");
      setAggregationType("terms");
      addAggregationsModalhandlers.close();
    }
  };

  return (
    <Modal
      opened={addAggregationsModalOpened}
      onClose={addAggregationsModalhandlers.close}
      title="Add aggregation"
      size="md"
      centered
    >
      <Stack p="md">
        <NativeSelect
          value={aggregationType}
          label="Aggregation Type"
          onChange={(event) => setAggregationType(event.currentTarget.value)}
          data={[
            { label: "Terms", value: "terms" },
            { label: "Cardinality", value: "cardinality" },
          ]}
        />
        <Select
          value={aggregationValue}
          label="Aggregation field"
          onChange={setAggregationValue}
          placeholder="Select field"
          data={fieldsData}
          searchable
          withinPortal
          // creatable
          getCreateLabel={(query) => `+ Aggregation on ${query}`}
        />
        <Group position="right">
          <Button bg="#5688b0" onClick={(e) => handleSubmit(e)}>
            Add Aggregation
          </Button>
        </Group>
      </Stack>
    </Modal>
  );
};
