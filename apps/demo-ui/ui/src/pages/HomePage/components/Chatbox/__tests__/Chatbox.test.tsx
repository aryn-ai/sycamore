import { ChatBox, LoadingChatBox } from "../index";
import { render, screen, waitFor, userEvent } from "../../../../../test-utils";
import "@testing-library/jest-dom";
import { Settings, SystemChat } from "../../../../../Types";
import fetchMock from "jest-fetch-mock";
import {
  createFeedbackIndex,
  getEmbeddingModels,
  getIndices,
  openSearchNoBodyCall,
} from "../../../../../utils/OpenSearch";

fetchMock.enableMocks();

window.scrollTo = jest.fn();

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

// jest.mock("../../../../../utils/OpenSearch", () => ({
//   createConversation: jest.fn(),
//   getIndices: jest.fn().mockResolvedValue({ sampleKey: "sampleValue" }),
//   getEmbeddingModels: jest.fn(),
//   ...jest.requireActual("../../../../../utils/OpenSearch"),
// }));

const mockProps = {
  chatHistory: [],
  searchResults: [],
  setChatHistory: jest.fn(),
  setSearchResults: jest.fn(),
  streaming: false,
  setStreaming: jest.fn(),
  setDocsLoading: jest.fn(),
  setErrorMessage: jest.fn(),
  settings: new Settings({
    auto_filter: false,
    required_filters: [],
    openSearchIndex: "mockIndex",
  }),
  setSettings: jest.fn(),
  refreshConversations: jest.fn(),
  chatInputRef: { current: null },
  openErrorDialog: jest.fn(),
};

describe("ChatBox Component", () => {
  it("renders the ChatBox container and basic elements", () => {
    render(<ChatBox {...mockProps} />);

    expect(screen.getByPlaceholderText("Ask me anything")).toBeInTheDocument();
    expect(screen.getByText("How can I help you today?")).toBeInTheDocument();
    expect(screen.getByRole("img")).toBeInTheDocument();
    expect(screen.getByTestId("rewrite-button")).toBeInTheDocument();
    expect(screen.getByTestId("settings-button")).toBeInTheDocument();
    expect(screen.queryByTestId("send-button")).not.toBeInTheDocument();
    // expect(getIndices).toHaveBeenCalled();
    // expect(getEmbeddingModels).toHaveBeenCalled();
  });

  it("renders the send button when the input field is filled", () => {
    render(<ChatBox {...mockProps} />);
    const inputField = screen.getByPlaceholderText("Ask me anything");
    expect(screen.queryByTestId("send-button")).not.toBeInTheDocument();
    userEvent.type(inputField, "Test Question");
    expect(inputField).toHaveValue("Test Question");
    expect(screen.queryByTestId("send-button")).toBeInTheDocument();
  });

  it("renders the ScrollArea and chat messages when chat history is not empty", () => {
    const chatHistory = [
      new SystemChat({ id: "1", response: "Hello", queryUsed: "Hi" }),
    ];
    render(<ChatBox {...mockProps} chatHistory={chatHistory} />);
    expect(screen.getByText("Hello")).toBeInTheDocument();
  });

  it("renders the LoadingChatBox when loadingMessage is set", async () => {
    render(<LoadingChatBox loadingMessage="Loading..." />);
    expect(screen.getByText("Loading...")).toBeInTheDocument();
  });

  it("render elements when auto-filter is enabled", () => {
    render(
      <ChatBox
        {...mockProps}
        settings={
          new Settings({
            openSearchIndex: "mockIndex",
            auto_filter: true,
            required_filters: [],
          })
        }
      />,
    );
    const addButtons = screen.getAllByText("+ Add");
    expect(addButtons).toHaveLength(2);
    addButtons.forEach((button) => {
      expect(button).toBeInTheDocument();
    });
    expect(screen.getByText(/Filters/i)).toBeInTheDocument();
    expect(screen.getByText(/Aggregations/i)).toBeInTheDocument();
  });
});
