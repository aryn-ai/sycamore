import {
  Button,
  Group,
  Modal,
  NativeSelect,
  SegmentedControl,
  Stack,
  TextInput,
} from "@mantine/core";
import { useState } from "react";

export const AddAggregationModal = ({
  addAggregationsModalOpened,
  addAggregationsModalhandlers,
  aggregations,
  setAggregations,
}: {
  addAggregationsModalOpened: boolean;
  addAggregationsModalhandlers: any;
  aggregations: any;
  setAggregations: any;
}) => {
  const [aggregationType, setAggregationType] = useState("terms");
  const [aggregationValue, setAggregationValue] = useState("location");

  const handleSubmit = (e: React.FormEvent) => {
    setAggregations((prevAggs: any) => ({
      [aggregationType]: aggregationValue,
    }));
    setAggregationValue("location");
    setAggregationType("terms");
    addAggregationsModalhandlers.close();
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
        <NativeSelect
          value={aggregationValue}
          label="Aggregation field"
          onChange={(event) => setAggregationValue(event.currentTarget.value)}
          data={[
            { value: "location", label: "Location" },
            { value: "aircraftType", label: "Aircraft type" },
          ]}
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
