import { render, screen, userEvent, waitFor } from "../../../../../test-utils";
import { AddAggregationModal } from "../AddAggregationModal";
import "@testing-library/jest-dom";

jest.mock("@mantine/hooks", () => ({
  ...jest.requireActual("@mantine/hooks"),
  useMediaQuery: jest.fn(),
  ResizeObserver: jest.fn(),
}));

global.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};

describe("AddAggregationModal", () => {
  const defaultProps = {
    addAggregationsModalOpened: true,
    addAggregationsModalhandlers: { close: jest.fn() },
    aggregations: {},
    setAggregations: jest.fn(),
    aggregationFields: [],
  };
  it("renders modal with correct title", () => {
    render(<AddAggregationModal {...defaultProps} />);

    expect(
      screen.getByRole("heading", { level: 2, name: /Add aggregation/i }),
    ).toBeInTheDocument();
  });

  it("renders aggregation type dropdown with default value", () => {
    render(<AddAggregationModal {...defaultProps} />);

    const aggregationTypeDropdown = screen.getByLabelText("Aggregation Type");
    expect(aggregationTypeDropdown).toBeInTheDocument();
    expect(aggregationTypeDropdown).toHaveValue("terms");
  });

  it("renders aggregation field select with options", () => {
    const aggregationFields = ["field1", "field2"];

    render(
      <AddAggregationModal
        {...defaultProps}
        aggregationFields={aggregationFields}
      />,
    );
    const aggregationFieldSelect = screen.getByLabelText("Aggregation field");
    expect(aggregationFieldSelect).toBeInTheDocument();
    userEvent.click(aggregationFieldSelect);
    expect(screen.getByText("field1")).toBeInTheDocument();
    expect(screen.getByText("field2")).toBeInTheDocument();
  });

  it("adds aggregation and closes modal on submit", async () => {
    const setAggregationsMock = jest.fn();
    const closeModalMock = jest.fn();

    render(
      <AddAggregationModal
        {...defaultProps}
        addAggregationsModalhandlers={{ close: closeModalMock }}
        setAggregations={setAggregationsMock}
        aggregationFields={["field1"]}
      />,
    );

    const aggregationTypeDropdown = screen.getByLabelText("Aggregation Type");
    userEvent.selectOptions(aggregationTypeDropdown, "cardinality");
    const aggregationFieldSelect = screen.getByLabelText("Aggregation field");

    userEvent.click(aggregationFieldSelect);
    userEvent.click(screen.getByText("field1"));
    const addButton = screen.getByRole("button", { name: /Add Aggregation/i });
    userEvent.click(addButton);

    await waitFor(() => {
      expect(setAggregationsMock).toHaveBeenCalledTimes(1);
    });
    expect(closeModalMock).toHaveBeenCalledTimes(1);
  });

  it("does not add aggregation when aggregation value is empty", async () => {
    const setAggregationsMock = jest.fn();

    render(
      <AddAggregationModal
        {...defaultProps}
        setAggregations={setAggregationsMock}
      />,
    );

    const aggregationFieldSelect = screen.getByLabelText("Aggregation field");
    expect(aggregationFieldSelect).toBeInTheDocument();

    const addButton = screen.getByRole("button", { name: /Add Aggregation/i });
    userEvent.click(addButton);

    await waitFor(() => {
      expect(setAggregationsMock).not.toHaveBeenCalled();
    });
  });
});
