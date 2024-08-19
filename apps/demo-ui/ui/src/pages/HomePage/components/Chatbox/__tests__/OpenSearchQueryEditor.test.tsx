import { render, screen, userEvent } from "../../../../../test-utils";
import { OpenSearchQueryEditor } from "../OpenSearchQueryEditor";
import "@testing-library/jest-dom";

jest.mock("../../../../../utils/OpenSearch", () => ({
  openSearchCall: jest.fn().mockResolvedValue({}),
}));
jest.mock("../../../../../utils/ChatboxUtils", () => ({
  interpretOsResult: jest.fn(),
}));

jest.mock("@mantine/hooks", () => ({
  ...jest.requireActual("@mantine/hooks"),
  ResizeObserver: jest.fn(),
}));

global.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};

const mockProps = {
  currentOsUrl: "http://example.com",
  openSearchQueryEditorOpened: true,
  openSearchQueryEditorOpenedHandlers: {
    close: jest.fn(),
  },
  currentOsQuery: '{"query": {}}',
  setCurrentOsQuery: jest.fn(),
  setCurrentOsUrl: jest.fn(),
  setLoadingMessage: jest.fn(),
  chatHistory: [],
  setChatHistory: jest.fn(),
};

describe("OpenSearchQueryEditor", () => {
  it("renders the modal with the correct title", () => {
    render(<OpenSearchQueryEditor {...mockProps} />);
    expect(screen.getByText("OpenSearch Query Editor")).toBeInTheDocument();
  });

  it("renders the OpenSearch URL input field with the correct placeholder and initial value", () => {
    render(<OpenSearchQueryEditor {...mockProps} />);
    const urlInput = screen.getByLabelText(/OpenSearch url/i);
    expect(urlInput).toBeInTheDocument();
    expect(urlInput).toHaveValue("http://example.com");
  });

  it("renders the JsonInput component with the correct placeholder and initial value", () => {
    render(<OpenSearchQueryEditor {...mockProps} />);
    const jsonInput = screen.getByTestId("json-input");
    expect(jsonInput).toBeInTheDocument();
    expect(jsonInput).toHaveValue('{"query": {}}');
  });

  it('renders the "Run" button', () => {
    render(<OpenSearchQueryEditor {...mockProps} />);
    const runButton = screen.getByText("Run");
    expect(runButton).toBeInTheDocument();
  });

  it("renders the informational text about RAG answers", () => {
    render(<OpenSearchQueryEditor {...mockProps} />);
    const infoText = screen.getByText(
      /Note: If you want a RAG answer, make sure the search pipeline is being used. Ensure it's configured in the URL/i,
    );
    expect(infoText).toBeInTheDocument();
  });

  it("updates currentOsUrl when the input value changes", () => {
    render(<OpenSearchQueryEditor {...mockProps} />);
    const urlInput = screen.getByLabelText(/OpenSearch url/i);
    userEvent.type(urlInput, "http://new-url.com");
    expect(mockProps.setCurrentOsUrl).toHaveBeenCalled();
  });

  it('calls handleOsSubmit when the "Run" button is clicked', () => {
    render(<OpenSearchQueryEditor {...mockProps} />);
    const runButton = screen.getByText("Run");
    userEvent.click(runButton);
    expect(mockProps.setLoadingMessage).toHaveBeenCalledWith(
      "Processing query...",
    );
  });
});
