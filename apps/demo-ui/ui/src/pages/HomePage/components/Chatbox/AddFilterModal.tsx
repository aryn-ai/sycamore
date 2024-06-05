import {
  ActionIcon,
  Badge,
  Button,
  Group,
  Modal,
  NativeSelect,
  Stack,
  Text,
  TextInput,
  rem,
} from "@mantine/core";
import { useEffect, useState } from "react";
import { DateInput } from "@mantine/dates";
import { IconX } from "@tabler/icons-react";

export const AddFilterModal = ({
  addFilterModalOpened,
  addFilterModalHandlers,
  filterContent,
  setFilterContent,
}: {
  addFilterModalOpened: boolean;
  addFilterModalHandlers: any;
  filterContent: any;
  setFilterContent: any;
}) => {
  const [newFilterType, setNewFilterType] = useState("location");
  const [newFilterValue, setNewFilterValue] = useState("");
  const [newDateValue, setNewDateValue] = useState<Date | null>(null);
  const [newFilterContent, setNewFilterContent] = useState({});

  useEffect(() => {
    setNewFilterContent(filterContent);
  }, [filterContent]);

  const handleAdd = (e: React.FormEvent) => {
    setNewFilterContent((prevFilters: any) => ({
      ...prevFilters,
      [newFilterType]:
        newFilterType === "day_end" || newFilterType === "day_start"
          ? newDateValue?.toISOString().split("T")[0]
          : newFilterValue,
    }));
    setNewFilterValue("");
    setNewDateValue(null);
  };

  const handleRemoveFilter = (filterKeyToRemove: string) => {
    setNewFilterContent((prevFilters: any) => {
      const updatedFilters = { ...prevFilters };
      delete updatedFilters[filterKeyToRemove];
      return updatedFilters;
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    setFilterContent(newFilterContent);
    setNewFilterContent({});
    setNewFilterValue("");
    setNewDateValue(null);
    setNewFilterType("location");
    addFilterModalHandlers.close();
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setNewFilterValue(e.target.value);
  };

  const handleDateChange = (date: Date | null) => {
    setNewDateValue(date);
  };

  const handleInputKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleAdd(e);
    }
  };

  const FilterBadge = ({
    filterKey,
    filterValue,
  }: {
    filterKey: string;
    filterValue: any;
  }) => {
    if (filterKey === "day_start") {
      filterValue = "> " + filterValue;
    }
    if (filterKey === "day_end") {
      filterValue = "< " + filterValue;
    }
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
        {filterValue}
      </Badge>
    );
  };

  return (
    <Modal
      opened={addFilterModalOpened}
      onClose={addFilterModalHandlers.close}
      title="Add filter"
      size="md"
      centered
    >
      <Stack p="md">
        <NativeSelect
          value={newFilterType}
          label="Field to filter on"
          onChange={(event) => setNewFilterType(event.currentTarget.value)}
          data={[
            { value: "location", label: "Location" },
            { value: "aircraftType", label: "Airplane type" },
            { value: "day_end", label: "Before" },
            { value: "day_start", label: "After" },
          ]}
        />
        {newFilterType === "day_end" || newFilterType === "day_start" ? (
          <DateInput
            value={newDateValue}
            onChange={handleDateChange}
            label="Date input"
            placeholder="Date input"
            clearable
            popoverProps={{ withinPortal: true }}
            valueFormat="YYYY/MM/DD"
          />
        ) : (
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
        )}
        {Object.keys(newFilterContent).length !== 0 && (
          <Group spacing="xs">
            <Text size="xs">Filters :</Text>
            {Object.entries(newFilterContent).map(([key, value]) => (
              <FilterBadge key={key} filterKey={key} filterValue={value} />
            ))}
          </Group>
        )}

        <Group position="right">
          <Button bg="#5688b0" onClick={(e) => handleAdd(e)}>
            Add filter
          </Button>
          <Button
            bg="#5688b0"
            onClick={(e) => handleSubmit(e)}
            disabled={Object.keys(newFilterContent).length === 0}
          >
            Apply
          </Button>
        </Group>
      </Stack>
    </Modal>
  );
};
