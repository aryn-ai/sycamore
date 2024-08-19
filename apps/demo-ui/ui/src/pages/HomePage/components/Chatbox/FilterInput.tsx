import { Group, Text, TextInput } from "@mantine/core";
import { Settings } from "../../../../Types";

export const FilterInput = ({
  settings,
  filtersInput,
  setFiltersInput,
  filterError,
  setFilterError,
}: {
  settings: Settings;
  filtersInput: any;
  setFiltersInput: any;
  filterError: boolean;
  setFilterError: any;
}) => {
  const handleInputChange = (filterName: string, value: string) => {
    if (filterError) {
      setFilterError(false);
    }
    setFiltersInput((prevValues: any) => ({
      ...prevValues,
      [filterName]: value,
    }));
  };

  return (
    <Group spacing="0">
      {settings.required_filters.map((required_filter) => (
        <Group spacing="0" key={required_filter}>
          <Text size="xs">{required_filter}</Text>
          <TextInput
            data-testid={`${required_filter}-input`}
            onChange={(e) => handleInputChange(required_filter, e.target.value)}
            value={filtersInput[required_filter] || ""}
            autoFocus
            required
            error={filterError}
            size="xs"
            fz="xs"
            pl="sm"
          />
        </Group>
      ))}
    </Group>
  );
};
