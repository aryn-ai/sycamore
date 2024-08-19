import { ControlPanel } from "../Controlpanel";
import { Settings } from "../../../../../Types";
import {
  getIndices,
  getEmbeddingModels,
} from "../../../../../utils/OpenSearch";
import { render, screen, userEvent, waitFor } from "../../../../../test-utils";
import "@testing-library/jest-dom";

jest.mock("../../../../../utils/OpenSearch");

const mockSettings: Settings = new Settings({
  ragPassageCount: 3,
  modelName: "test-model",
  openSearchIndex: "test-index",
  embeddingModel: "test-embedding",
  availableModels: ["test-model", "another-model"],
});

const mockSetSettings = jest.fn();
const mockOnControlPanelClose = jest.fn();
const mockOpenErrorDialog = jest.fn();

describe("ControlPanel", () => {
  beforeEach(() => {
    Storage.prototype.setItem = jest.fn();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it("renders the modal with the correct title", () => {
    render(
      <ControlPanel
        settings={mockSettings}
        setSettings={mockSetSettings}
        controlPanelOpened={true}
        onControlPanelClose={mockOnControlPanelClose}
        openErrorDialog={mockOpenErrorDialog}
      />,
    );
    expect(screen.getByText("Options")).toBeInTheDocument();
  });

  it("renders form elements with correct initial values from settings", async () => {
    (getIndices as jest.Mock).mockResolvedValue({ "test-index": {} });
    (getEmbeddingModels as jest.Mock).mockResolvedValue({
      hits: { hits: [{ _id: "test-embedding" }] },
    });
    render(
      <ControlPanel
        settings={mockSettings}
        setSettings={mockSetSettings}
        controlPanelOpened={true}
        onControlPanelClose={mockOnControlPanelClose}
        openErrorDialog={mockOpenErrorDialog}
      />,
    );
    expect(screen.getByDisplayValue("3")).toBeInTheDocument();
    expect(screen.getByDisplayValue("test-model")).toBeInTheDocument();

    // expect(screen.getByDisplayValue("test-index")).toBeInTheDocument();
    // expect(screen.getByDisplayValue("test-embedding")).toBeInTheDocument();
  });

  it("fetches indices and embedding models on initial render", async () => {
    (getIndices as jest.Mock).mockResolvedValue({ "test-index": {} });
    (getEmbeddingModels as jest.Mock).mockResolvedValue({
      hits: { hits: [{ _id: "test-embedding" }] },
    });

    render(
      <ControlPanel
        settings={mockSettings}
        setSettings={mockSetSettings}
        controlPanelOpened={true}
        onControlPanelClose={mockOnControlPanelClose}
        openErrorDialog={mockOpenErrorDialog}
      />,
    );

    await waitFor(() => {
      expect(getIndices).toHaveBeenCalled();
      expect(getEmbeddingModels).toHaveBeenCalled();
    });
  });

  it('renders "Set as default" checkbox and "Submit" button', () => {
    render(
      <ControlPanel
        settings={mockSettings}
        setSettings={mockSetSettings}
        controlPanelOpened={true}
        onControlPanelClose={mockOnControlPanelClose}
        openErrorDialog={mockOpenErrorDialog}
      />,
    );
    expect(screen.getByLabelText("Set as default")).toBeInTheDocument();
    expect(screen.getByText("Submit")).toBeInTheDocument();
  });

  it("updates formValues when input values change", () => {
    render(
      <ControlPanel
        settings={mockSettings}
        setSettings={mockSetSettings}
        controlPanelOpened={true}
        onControlPanelClose={mockOnControlPanelClose}
        openErrorDialog={mockOpenErrorDialog}
      />,
    );

    userEvent.selectOptions(screen.getByDisplayValue("3"), "5");
    expect(screen.getByDisplayValue("5")).toBeInTheDocument();

    userEvent.selectOptions(
      screen.getByDisplayValue("test-model"),
      "another-model",
    );
    expect(screen.getByDisplayValue("another-model")).toBeInTheDocument();
  });
  it("updates isDefault when checkbox is toggled", () => {
    render(
      <ControlPanel
        settings={mockSettings}
        setSettings={mockSetSettings}
        controlPanelOpened={true}
        onControlPanelClose={mockOnControlPanelClose}
        openErrorDialog={mockOpenErrorDialog}
      />,
    );

    const checkbox = screen.getByLabelText("Set as default");
    userEvent.click(checkbox);
    expect(checkbox).toBeChecked();
  });

  it("calls handleSubmit on form submission", () => {
    render(
      <ControlPanel
        settings={mockSettings}
        setSettings={mockSetSettings}
        controlPanelOpened={true}
        onControlPanelClose={mockOnControlPanelClose}
        openErrorDialog={mockOpenErrorDialog}
      />,
    );

    const form = screen.getByRole("button", { name: /submit/i });
    userEvent.click(form);
    expect(mockSetSettings).toHaveBeenCalled();
    expect(mockOnControlPanelClose).toHaveBeenCalled();
  });

  it('saves settings to localStorage when "Set as default" is checked and form is submitted', () => {
    render(
      <ControlPanel
        settings={mockSettings}
        setSettings={mockSetSettings}
        controlPanelOpened={true}
        onControlPanelClose={mockOnControlPanelClose}
        openErrorDialog={mockOpenErrorDialog}
      />,
    );

    const checkbox = screen.getByLabelText("Set as default");
    userEvent.click(checkbox);
    const form = screen.getByRole("button", { name: /submit/i });
    userEvent.click(form);

    expect(localStorage.setItem).toHaveBeenCalledWith(
      "defaultSettings",
      JSON.stringify({
        ragPassageCount: 3,
        modelName: "test-model",
        openSearchIndex: "test-index",
        embeddingModel: "test-embedding",
      }),
    );
  });

  it("handles errors during initial loading or reloading of indices and embeddings", async () => {
    (getIndices as jest.Mock).mockRejectedValue(
      new Error("Failed to fetch indices"),
    );
    (getEmbeddingModels as jest.Mock).mockRejectedValue(
      new Error("Failed to fetch embeddings"),
    );

    render(
      <ControlPanel
        settings={mockSettings}
        setSettings={mockSetSettings}
        controlPanelOpened={true}
        onControlPanelClose={mockOnControlPanelClose}
        openErrorDialog={mockOpenErrorDialog}
      />,
    );

    await waitFor(() => {
      expect(mockOpenErrorDialog).toHaveBeenCalledTimes(2);
    });
  });
});
