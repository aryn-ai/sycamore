import { render, screen, userEvent, waitFor } from "../../../../../test-utils";
import { Settings, SystemChat } from "../../../../../Types";
import {
  getHybridConversationSearchQuery,
  hybridConversationSearch,
  updateInteractionAnswer,
} from "../../../../../utils/OpenSearch";
import { SystemChatBox } from "../SystemChatbox";
import "@testing-library/jest-dom";

jest.mock("@mantine/hooks", () => ({
  ...jest.requireActual("@mantine/hooks"),
  ResizeObserver: jest.fn(),
}));

jest.mock("../../../../../utils/OpenSearch", () => ({
  hybridConversationSearch: jest.fn().mockResolvedValue([
    Promise.resolve({
      ext: {
        retrieval_augmented_generation: {
          answer: "Mocked answer",
          interaction_id: "mocked_interaction_id",
        },
      },
      hits: {
        hits: [],
      },
    }),
    {},
  ]),
  hybridSearchNoRag: jest.fn().mockResolvedValue({}),
  getHybridConversationSearchQuery: jest.fn().mockReturnValue({
    query: {},
    url: "",
  }),
  updateInteractionAnswer: jest.fn().mockResolvedValue({}),
}));

global.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};

const mockSystemChat: SystemChat = new SystemChat({
  id: "1",
  queryUsed: "Who are you?",
  response: "I am you.",
  hits: [],
  rawResults: null,
  originalQuery: "",
  interaction_id: "12345",
  filterContent: {
    location: "California",
    airplane_name: "Boeing 747",
  },
  aggregationContent: {
    day_start: "2021-01-01",
    day_end: "2021-12-31",
  },
});

const mockSettings: Settings = new Settings({
  auto_filter: false,
  required_filters: ["Filter1", "Filter2"],
});

const mockProps = {
  systemChat: mockSystemChat,
  settings: mockSettings,
  handleSubmit: jest.fn(),
  setCurrentOsQuery: jest.fn(),
  setCurrentOsUrl: jest.fn(),
  openSearchQueryEditorOpenedHandlers: {
    open: jest.fn(),
    close: jest.fn(),
  },
  disableFilters: false,
  setSearchResults: jest.fn(),
  setErrorMessage: jest.fn(),
  setLoadingMessage: jest.fn(),
  setManualAggregations: jest.fn(),
  setChatHistory: jest.fn(),
  chatHistory: [],
  anthropicRagFlag: false,
  streamingRagResponse: false,
  setStreamingRagResponse: jest.fn(),
  openErrorDialog: jest.fn(),
  setManualFilters: jest.fn(),
};

describe("SystemChatBox", () => {
  it("renders the system chat message correctly", () => {
    render(<SystemChatBox {...mockProps} />);
    expect(screen.getByText("I am you.")).toBeInTheDocument();
  });

  it('renders the "OpenSearch results" section when rawResults are available', () => {
    const systemChatWithResults = {
      ...mockSystemChat,
      rawResults: { hits: { hits: [] } },
    };
    render(<SystemChatBox {...mockProps} systemChat={systemChatWithResults} />);
    expect(screen.getByText("OpenSearch results")).toBeInTheDocument();
  });

  it("renders the DocList component with the correct documents", () => {
    const systemChatWithDocs = {
      ...mockSystemChat,
      hits: [
        {
          id: "1",
          title: "Test Document",
          url: "/test-document.pdf",
          description: "This is a test document description.",
          properties: {
            entity: {
              location: "Test Location",
              aircraftType: "Test Aircraft",
              day: "Test Day",
            },
          },
          index: 1,
          isPdf: () => true,
          hasAbsoluteUrl: () => false,
          relevanceScore: "0.9",
          bbox: [0, 0, 100, 100],
        },
      ],
    };
    render(<SystemChatBox {...mockProps} systemChat={systemChatWithDocs} />);
    expect(screen.getByText("Test Document")).toBeInTheDocument();
  });

  it("renders the original query if it is different from the query used", () => {
    const systemChatWithOriginalQuery = {
      ...mockSystemChat,
      originalQuery: "Who am I?",
    };
    render(
      <SystemChatBox {...mockProps} systemChat={systemChatWithOriginalQuery} />,
    );
    expect(screen.getByText("Original Query: Who am I?")).toBeInTheDocument();
  });

  it("renders the interaction ID if available", () => {
    render(<SystemChatBox {...mockProps} />);
    expect(screen.getByText("Interaction id: 12345")).toBeInTheDocument();
  });

  it("renders the FeedbackButtons component", () => {
    render(<SystemChatBox {...mockProps} />);
    expect(screen.getByTestId("thumb-up-icon")).toBeInTheDocument();
    expect(screen.getByTestId("thumb-down-icon")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Leave a comment")).toBeInTheDocument();
  });

  it("renders the edit icon when not in editing mode", () => {
    render(<SystemChatBox {...mockProps} />);
    expect(screen.getByTestId("edit-icon")).toBeInTheDocument();
  });

  //   it("renders filters if auto_filter is true and filters are present", () => {
  //     render(<SystemChatBox {...mockProps} />);
  //     expect(screen.getByText("Filters")).toBeInTheDocument();
  //     mockSettings.required_filters.forEach((filter) => {
  //       expect(screen.getByText(filter)).toBeInTheDocument();
  //     });
  //   });

  //   it("renders aggregations if auto_filter is true and aggregations are present", () => {
  //     render(<SystemChatBox {...mockProps} />);
  //     expect(screen.getByText("Aggregations")).toBeInTheDocument();
  //   });

  //   it("renders the loading message when loadingMessage is set", () => {
  //     render(<SystemChatBox {...mockProps} />);
  //     expect(screen.getByText("Loading...")).toBeInTheDocument();
  //   });

  it('toggles the OpenSearch results section when the "OpenSearch results" header is clicked', () => {
    const systemChatWithResults = {
      ...mockSystemChat,
      rawResults: { hits: { hits: [] } },
    };
    render(<SystemChatBox {...mockProps} systemChat={systemChatWithResults} />);
    const header = screen.getByText("OpenSearch results");
    userEvent.click(header);
    expect(screen.getByText("OpenSearch results")).toBeInTheDocument();
  });

  it("switches to edit mode when the edit icon is clicked", () => {
    render(<SystemChatBox {...mockProps} />);
    userEvent.click(screen.getByTestId("edit-icon"));
    expect(screen.getByTestId("edit-input")).toBeInTheDocument();
  });

  it("updates newQuestion on input change in edit mode", () => {
    render(<SystemChatBox {...mockProps} />);
    userEvent.click(screen.getByTestId("edit-icon"));
    const input = screen.getByTestId("edit-input");
    userEvent.type(input, " Really?");
    expect(input).toHaveValue("Who are you? Really?");
  });

  it("exits edit mode and resets newQuestion when the close icon is clicked", async () => {
    render(<SystemChatBox {...mockProps} />);
    userEvent.click(screen.getByTestId("edit-icon"));
    const input = screen.getByTestId("edit-input");
    userEvent.type(input, "New question?");
    userEvent.click(screen.getByTestId("close-icon"));

    expect(screen.queryByTestId("edit-input")).not.toBeInTheDocument();
  });

  it('opens the OpenSearch query editor when the "OpenSearch query editor" button is clicked', () => {
    render(<SystemChatBox {...mockProps} />);
    userEvent.click(screen.getByTestId("edit-icon"));
    userEvent.click(screen.getByText("OpenSearch query editor"));
    expect(mockProps.setCurrentOsQuery).toHaveBeenCalled();
    expect(mockProps.setCurrentOsUrl).toHaveBeenCalled();
  });

  // it("calls setManualFilters when the copy icon next to filters is clicked", () => {
  //   const setManualFiltersMock = jest.fn();
  //   render(
  //     <SystemChatBox {...mockProps} setManualFilters={setManualFiltersMock} />,
  //   );
  //   userEvent.click(screen.getByTestId("copy-filters-icon"));
  //   expect(setManualFiltersMock).toHaveBeenCalled();
  // });

  // it("calls setManualAggregations when the copy icon next to aggregations is clicked", () => {
  //   const setManualAggregationsMock = jest.fn();
  //   render(
  //     <SystemChatBox
  //       {...mockProps}
  //       setManualAggregations={setManualAggregationsMock}
  //     />,
  //   );
  //   userEvent.click(screen.getByTestId("copy-aggregations-icon"));
  //   expect(setManualAggregationsMock).toHaveBeenCalled();
  // });
});
