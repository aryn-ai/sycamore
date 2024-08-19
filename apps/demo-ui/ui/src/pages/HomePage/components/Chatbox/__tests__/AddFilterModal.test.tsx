import { render, screen, userEvent } from "../../../../../test-utils";
import { AddFilterModal } from "../AddFilterModal";
import { FilterValues } from "../../../../../Types";
import "@testing-library/jest-dom";

jest.mock("@mantine/hooks", () => ({
  ...jest.requireActual("@mantine/hooks"),
  ResizeObserver: jest.fn(),
}));

global.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};

describe("AddFilterModal", () => {
  const defaultProps = {
    addFilterModalOpened: true,
    addFilterModalHandlers: { close: jest.fn() },
    filterContent: {} as FilterValues,
    setFilterContent: jest.fn(),
    filterFields: ["field1", "field2"],
  };

  it("renders the modal with the correct title and field select", () => {
    render(<AddFilterModal {...defaultProps} />);
    expect(
      screen.getByRole("heading", { level: 2, name: /Add filter/i }),
    ).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Select field")).toBeInTheDocument();
  });

  it("renders filter badges when newFilterContent is not empty", () => {
    const filterContent = { field1: "value1" };
    render(<AddFilterModal {...defaultProps} filterContent={filterContent} />);
    expect(screen.getByText("field1: value1")).toBeInTheDocument();
  });

  it('renders "Add filter" and "Apply" buttons', () => {
    render(<AddFilterModal {...defaultProps} />);
    expect(
      screen.getByRole("button", { name: /Add filter/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Apply/i })).toBeInTheDocument();
  });

  it('disables the "Apply" button when newFilterContent is empty', () => {
    render(<AddFilterModal {...defaultProps} />);
    expect(screen.getByRole("button", { name: /Apply/i })).toBeDisabled();
  });

  it('enables the "Apply" button when newFilterContent is not empty', () => {
    const filterContent = { field1: "value1" };
    render(<AddFilterModal {...defaultProps} filterContent={filterContent} />);
    expect(screen.getByText("Apply")).not.toBeDisabled();
  });

  it('renders DateInput components when filter type is "day"', () => {
    render(<AddFilterModal {...defaultProps} />);
    const filterSelect = screen.getByLabelText("Field to filter on");
    userEvent.click(filterSelect);
    const dayOption = screen.getByText("Day");
    userEvent.click(dayOption);
    expect(screen.getByLabelText("Start")).toBeInTheDocument();
    expect(screen.getByLabelText("End")).toBeInTheDocument();
  });

  it("renders a single TextInput for other filter types", () => {
    const filterContent = {
      field3: "value1",
    };
    render(<AddFilterModal {...defaultProps} filterContent={filterContent} />);
    const filterSelect = screen.getByLabelText("Field to filter on");
    userEvent.click(filterSelect);
    const field3Option = screen.getByText(/field3/i);
    userEvent.click(field3Option);
    expect(screen.getByText(/field3/i)).toBeInTheDocument();
    expect(screen.getByLabelText("Value of filter")).toBeInTheDocument();
  });

  it("renders TextInput components for range filter type", () => {
    render(<AddFilterModal {...defaultProps} />);
    const filterSelect = screen.getByLabelText("Field to filter on");
    userEvent.click(filterSelect);
    const rangeOption = screen.getByText("Wind Speed in knots");
    userEvent.click(rangeOption);
    expect(screen.getByLabelText("Minimum")).toBeInTheDocument();
    expect(screen.getByLabelText("Maximum")).toBeInTheDocument();
  });

  it('adds a new range filter to newFilterContent on "Add filter" button click', () => {
    render(<AddFilterModal {...defaultProps} />);
    const filterSelect = screen.getByLabelText("Field to filter on");
    userEvent.click(filterSelect);
    const dayOption = screen.getByText("Day");
    userEvent.click(dayOption);

    const startDateInput = screen.getByLabelText("Start");
    const endDateInput = screen.getByLabelText("End");
    userEvent.type(startDateInput, "2023/07/20");
    userEvent.type(endDateInput, "2023/07/30");
    const addButton = screen.getByRole("button", { name: /Add filter/i });
    userEvent.click(addButton);
    expect(screen.getByText(/2023-07-20 - 2023-07-30/i)).toBeInTheDocument();
  });

  it('adds a new value filter to newFilterContent on "Add filter" button click', () => {
    render(<AddFilterModal {...defaultProps} />);
    const filterSelect = screen.getByLabelText("Field to filter on");
    userEvent.click(filterSelect);
    const locationOption = screen.getByText("Location");
    userEvent.click(locationOption);

    const filterInput = screen.getByLabelText("Value of filter");
    userEvent.type(filterInput, "California");
    const addButton = screen.getByRole("button", { name: /Add filter/i });
    userEvent.click(addButton);
    expect(screen.getByText(/California/i)).toBeInTheDocument();
  });

  it("calls handleRemoveFilter when a filter badge is clicked", () => {
    render(<AddFilterModal {...defaultProps} />);
    const filterSelect = screen.getByLabelText("Field to filter on");
    userEvent.click(filterSelect);
    const locationOption = screen.getByText("Location");
    userEvent.click(locationOption);

    const filterInput = screen.getByLabelText("Value of filter");
    userEvent.type(filterInput, "California");
    const addButton = screen.getByRole("button", { name: /Add filter/i });
    userEvent.click(addButton);

    const removeButton = screen.getByTestId("remove-filter-button-location");
    userEvent.click(removeButton);
    expect(screen.queryByText(/California/i)).not.toBeInTheDocument();
  });

  it('calls setFilterContent and closes the modal on "Apply" button click', () => {
    const setFilterContent = jest.fn();
    const addFilterModalHandlers = { close: jest.fn() };
    render(
      <AddFilterModal
        {...defaultProps}
        setFilterContent={setFilterContent}
        addFilterModalHandlers={addFilterModalHandlers}
      />,
    );
    const filterSelect = screen.getByLabelText("Field to filter on");
    userEvent.click(filterSelect);
    const locationOption = screen.getByText("Location");
    userEvent.click(locationOption);

    const filterInput = screen.getByLabelText("Value of filter");
    userEvent.type(filterInput, "California");
    const addButton = screen.getByRole("button", { name: /Add filter/i });
    userEvent.click(addButton);
    const applyButton = screen.getByText("Apply");
    userEvent.click(applyButton);
    expect(setFilterContent).toHaveBeenCalled();
    expect(addFilterModalHandlers.close).toHaveBeenCalled();
  });

  it('calls setFilterContent and closes the modal on "Apply" button click', () => {
    const setFilterContent = jest.fn();
    const addFilterModalHandlers = { close: jest.fn() };
    render(
      <AddFilterModal
        {...defaultProps}
        setFilterContent={setFilterContent}
        addFilterModalHandlers={addFilterModalHandlers}
      />,
    );
    const filterSelect = screen.getByLabelText("Field to filter on");
    userEvent.click(filterSelect);
    const locationOption = screen.getByText("Location");
    userEvent.click(locationOption);

    const filterInput = screen.getByLabelText("Value of filter");
    userEvent.type(filterInput, "California");
    const addButton = screen.getByRole("button", { name: /Add filter/i });
    userEvent.click(addButton);
    const applyButton = screen.getByText("Apply");
    userEvent.click(applyButton);
    expect(setFilterContent).toHaveBeenCalled();
    expect(addFilterModalHandlers.close).toHaveBeenCalled();
  });
});
