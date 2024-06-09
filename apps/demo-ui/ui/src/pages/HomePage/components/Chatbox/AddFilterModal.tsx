import {
  ActionIcon,
  Badge,
  Button,
  Group,
  Modal,
  NativeSelect,
  Select,
  Stack,
  Text,
  TextInput,
  rem,
} from "@mantine/core";
import { useEffect, useState } from "react";
import { DateInput } from "@mantine/dates";
import { IconX } from "@tabler/icons-react";
import { FilterValues } from "../../../../Types";
import { documentLevelFields } from "../../../../utils/ChatboxUtils";

export const AddFilterModal = ({
  addFilterModalOpened,
  addFilterModalHandlers,
  filterContent,
  setFilterContent,
  filterFields,
}: {
  addFilterModalOpened: boolean;
  addFilterModalHandlers: any;
  filterContent: any;
  setFilterContent: any;
  filterFields: string[];
}) => {
  const [newFilterType, setNewFilterType] = useState<string | null>("");
  const [newFilterValue, setNewFilterValue] = useState("");
  const [startDateValue, setStartDateValue] = useState<Date | null>(null);
  const [endDateValue, setEndDateValue] = useState<Date | null>(null);
  const [newFilterContent, setNewFilterContent] = useState<FilterValues>({});
  const [minFilterValue, setMinFilterValue] = useState("");
  const [maxFilterValue, setMaxFilterValue] = useState("");
  const [fieldsData, setFieldsData] = useState<
    { value: string; label: string; group: string }[]
  >([]);

  useEffect(() => {
    setNewFilterContent(filterContent);
  }, [filterContent]);
  const rangeFilterFields = [
    "altimeterSettingInHg",
    "altimeterSettingInInchesHg",
    "dewPointTemperatureInC",
    "distanceFromAccidentSiteInNauticalMiles",
    "ratedPowerInHorsepower",
    "seats",
    "temperatureInC",
    "timeSinceLastInspectionInHrs",
    "visibilityInMiles",
    "windGustsInKnots",
    "windSpeedInKnots",
    "yearOfManufacture",
  ];

  useEffect(() => {
    const retrievedFieldsData = filterFields.map((field) => ({
      value: field,
      label: field,
    }));
    const originalFieldsData = [
      { value: "location", label: "Location" },
      { value: "aircraftType", label: "Aircraft type" },
      { value: "day", label: "Day" },
      { value: "lowestCloudCondition", label: "Lowest cloud condition" },
      { value: "windSpeedInKnots", label: "Wind Speed in knots" },
      { value: "injuries", label: "Injuries" },
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

  const handleAdd = (e: React.FormEvent) => {
    if (newFilterType === "day") {
      const existingFilters = { ...newFilterContent } as FilterValues;
      let dayValues: any = existingFilters.day || {};
      if (startDateValue) {
        dayValues["gte"] = startDateValue?.toISOString().split("T")[0];
      }
      if (endDateValue) {
        dayValues["lte"] = endDateValue?.toISOString().split("T")[0];
      }
      if (Object.keys(dayValues).length > 0) {
        setNewFilterContent((prevFilters: any) => ({
          ...prevFilters,
          day: dayValues,
        }));
      }
      setStartDateValue(null);
      setEndDateValue(null);
    } else if (
      newFilterType === "windSpeedInKnots" ||
      (newFilterType && rangeFilterFields.includes(newFilterType))
    ) {
      const existingFilters = { ...newFilterContent } as FilterValues;
      let filterValues: any = existingFilters[newFilterType] || {};
      if (minFilterValue !== "") {
        filterValues["gte"] = Number(minFilterValue);
      }
      if (maxFilterValue !== "") {
        filterValues["lte"] = Number(maxFilterValue);
      }
      if (Object.keys(filterValues).length > 0) {
        setNewFilterContent((prevFilters: any) => ({
          ...prevFilters,
          [newFilterType]: filterValues,
        }));
      }
      setMaxFilterValue("");
      setMinFilterValue("");
    } else if (
      newFilterType &&
      newFilterType.length !== 0 &&
      newFilterValue &&
      newFilterValue.length !== 0
    ) {
      setNewFilterContent((prevFilters: any) => ({
        ...prevFilters,
        [newFilterType as string]: newFilterValue,
      }));
      setNewFilterValue("");
    }
    console.log("inside filter modal on add: ", newFilterContent);
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
    // // setNewDateValue(null);
    // setNewFilterType("");
    addFilterModalHandlers.close();
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setNewFilterValue(e.target.value);
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

  return (
    <Modal
      opened={addFilterModalOpened}
      onClose={addFilterModalHandlers.close}
      title="Add filter"
      size="md"
      centered
    >
      <Stack p="md">
        <Select
          value={newFilterType}
          label="Field to filter on"
          placeholder="Select field"
          onChange={setNewFilterType}
          searchable
          getCreateLabel={(query) => `+ Filter on ${query}`}
          withinPortal
          data={fieldsData}
        />
        {newFilterType === "day" ? (
          <Group noWrap>
            <DateInput
              w="50%"
              value={startDateValue}
              onChange={setStartDateValue}
              label="Start"
              placeholder="Date input"
              clearable
              popoverProps={{ withinPortal: true }}
              valueFormat="YYYY/MM/DD"
            />
            <DateInput
              w="50%"
              value={endDateValue}
              onChange={setEndDateValue}
              label="End"
              placeholder="Date input"
              clearable
              popoverProps={{ withinPortal: true }}
              valueFormat="YYYY/MM/DD"
            />
          </Group>
        ) : newFilterType === "windSpeedInKnots" ||
          (newFilterType && rangeFilterFields.includes(newFilterType)) ? (
          <Group noWrap>
            <TextInput
              w="50%"
              label="Minimum"
              value={minFilterValue}
              // onKeyDown={(e) => setMinFilterValue(e.currentTarget.value)}
              onChange={(e) => setMinFilterValue(e.currentTarget.value)}
              autoFocus
              size="sm"
              fz="xs"
              placeholder="Enter min value"
            />
            <TextInput
              w="50%"
              label="Maximum"
              value={maxFilterValue}
              // onKeyDown={handleInputKeyPress}
              onChange={(e) => setMaxFilterValue(e.currentTarget.value)}
              autoFocus
              size="sm"
              fz="xs"
              placeholder="Enter max value"
            />
          </Group>
        ) : (
          <TextInput
            label="Value of filter"
            value={newFilterValue}
            onKeyDown={handleInputKeyPress}
            onChange={handleInputChange}
            autoFocus
            size="sm"
            fz="xs"
            placeholder="Enter value"
          />
        )}
        {Object.keys(newFilterContent).length !== 0 && (
          <Group spacing="xs">
            <Text size="xs">Filters :</Text>
            {Object.entries(newFilterContent).map(
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
