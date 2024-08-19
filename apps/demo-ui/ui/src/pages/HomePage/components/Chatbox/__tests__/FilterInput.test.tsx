import { FilterInput } from "../FilterInput";
import { Settings } from "../../../../../Types";
import { render, screen, userEvent } from "../../../../../test-utils";
import "@testing-library/jest-dom";

const mockSettings: Settings = new Settings({
  required_filters: ["filter1", "filter2"],
});

const mockSetFiltersInput = jest.fn();
const mockSetFilterError = jest.fn();

describe("FilterInput", () => {
  it("renders input fields for each required filter", () => {
    render(
      <FilterInput
        settings={mockSettings}
        filtersInput={{}}
        setFiltersInput={mockSetFiltersInput}
        filterError={false}
        setFilterError={mockSetFilterError}
      />,
    );

    mockSettings.required_filters.forEach((filter: string) => {
      expect(screen.getByText(filter)).toBeInTheDocument();
    });
  });

  it("displays the correct filter labels", () => {
    render(
      <FilterInput
        settings={mockSettings}
        filtersInput={{}}
        setFiltersInput={mockSetFiltersInput}
        filterError={false}
        setFilterError={mockSetFilterError}
      />,
    );

    mockSettings.required_filters.forEach((filter) => {
      expect(screen.getByText(filter)).toBeInTheDocument();
    });
  });

  it("updates filtersInput on input change", () => {
    render(
      <FilterInput
        settings={mockSettings}
        filtersInput={{}}
        setFiltersInput={mockSetFiltersInput}
        filterError={false}
        setFilterError={mockSetFilterError}
      />,
    );

    const input = screen.getByTestId("filter1-input");
    userEvent.type(input, "new");

    expect(mockSetFiltersInput).toHaveBeenCalledWith(expect.any(Function));
    expect(mockSetFiltersInput.mock.calls[0][0]({})).toEqual({
      filter1: "n",
    });
    expect(mockSetFiltersInput.mock.calls[1][0]({})).toEqual({
      filter1: "e",
    });
    expect(mockSetFiltersInput.mock.calls[2][0]({})).toEqual({
      filter1: "w",
    });
  });

  it("clears filterError when input changes", () => {
    render(
      <FilterInput
        settings={mockSettings}
        filtersInput={{}}
        setFiltersInput={mockSetFiltersInput}
        filterError={true}
        setFilterError={mockSetFilterError}
      />,
    );

    const input = screen.getByTestId("filter1-input");
    userEvent.type(input, "new value");

    expect(mockSetFilterError).toHaveBeenCalledWith(false);
  });
});
