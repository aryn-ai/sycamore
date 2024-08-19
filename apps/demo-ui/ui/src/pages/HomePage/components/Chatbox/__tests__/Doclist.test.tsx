import { render, screen, userEvent } from "../../../../../test-utils";
import "@testing-library/jest-dom";
import { DocList } from "../Doclist";
import { SearchResultDocument } from "../../../../../Types";
import { Settings } from "../../../../../Types";

jest.mock("@mantine/hooks", () => ({
  ...jest.requireActual("@mantine/hooks"),
  ResizeObserver: jest.fn(),
}));

global.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};

const mockDocuments: SearchResultDocument[] = [
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
  {
    id: "2",
    title: "Test Document 2",
    url: "/test-document-2.pdf",
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
  {
    id: "3",
    title: "Test Document 3",
    url: "http://example.com",
    description: "This is a test document description.",
    properties: {
      entity: {
        location: "Test Location",
        aircraftType: "Test Aircraft",
        day: "Test Day",
      },
    },
    index: 1,
    isPdf: () => false,
    hasAbsoluteUrl: () => false,
    relevanceScore: "0.9",
    bbox: [0, 0, 100, 100],
  },
  {
    id: "4",
    title: "Test Document 4",
    url: "http://example.com/document.html",
    description: "This is a test document description.",
    properties: {
      entity: {
        location: "Test Location",
        aircraftType: "Test Aircraft",
        day: "Test Day",
      },
    },
    index: 1,
    isPdf: () => false,
    hasAbsoluteUrl: () => false,
    relevanceScore: "0.9",
    bbox: [0, 0, 100, 100],
  },
];

const mockSettings = new Settings();

describe("DocList", () => {
  it("renders the document title correctly", () => {
    render(
      <DocList
        documents={mockDocuments}
        settings={mockSettings}
        docsLoading={false}
      />,
    );
    expect(screen.getByText("Test Document")).toBeInTheDocument();
    expect(screen.getByText("Test Document 2")).toBeInTheDocument();
    expect(screen.getByText("Test Document 3")).toBeInTheDocument();
    expect(screen.getByText("Test Document 4")).toBeInTheDocument();
  });

  it("renders the document filename correctly", () => {
    render(
      <DocList
        documents={mockDocuments}
        settings={mockSettings}
        docsLoading={false}
      />,
    );
    expect(screen.getByText("test-document.pdf")).toBeInTheDocument();
    expect(screen.getByText("test-document-2.pdf")).toBeInTheDocument();
    expect(screen.getByText("example.com")).toBeInTheDocument();
    expect(screen.getByText("document.html")).toBeInTheDocument();
  });

  it("renders a shortened snippet of the document description", () => {
    render(
      <DocList
        documents={mockDocuments}
        settings={mockSettings}
        docsLoading={false}
      />,
    );
    const documentTitle = screen.getByText("Test Document");
    userEvent.hover(documentTitle);
    expect(
      screen.getByText("This is a test document description."),
    ).toBeInTheDocument();
  });

  it("renders badges for specific entity properties if present (location, aircraftType, day)", () => {
    render(
      <DocList
        documents={mockDocuments}
        settings={mockSettings}
        docsLoading={false}
      />,
    );
    const documentTitle = screen.getByText("Test Document");
    userEvent.hover(documentTitle);
    expect(screen.getByText("location Test Location")).toBeInTheDocument();
    expect(screen.getByText("aircraftType Test Aircraft")).toBeInTheDocument();
    expect(screen.getByText("day Test Day")).toBeInTheDocument();
  });
  it("renders the correct icon based on document type (Link)", () => {
    render(
      <DocList
        documents={[mockDocuments[2]]}
        settings={mockSettings}
        docsLoading={false}
      />,
    );
    expect(screen.getByTestId("icon-link")).toBeInTheDocument();
  });

  it("renders the correct icon based on document type (PDF)", () => {
    render(
      <DocList
        documents={[mockDocuments[0]]}
        settings={mockSettings}
        docsLoading={false}
      />,
    );
    expect(screen.getByTestId("icon-pdf")).toBeInTheDocument();
  });

  it("renders the correct icon based on document type (HTML)", () => {
    render(
      <DocList
        documents={[mockDocuments[3]]}
        settings={mockSettings}
        docsLoading={false}
      />,
    );
    expect(screen.getByTestId("icon-html")).toBeInTheDocument();
  });

  it("opens the PDF viewer when a PDF document is clicked", () => {
    window.open = jest.fn();
    render(
      <DocList
        documents={[mockDocuments[0]]}
        settings={mockSettings}
        docsLoading={false}
      />,
    );
    const documentTitle = screen.getByText("Test Document");
    userEvent.click(documentTitle);
    expect(window.open).toHaveBeenCalledWith("/viewPdf");
  });

  it("opens the document URL in a new tab when a non-PDF document is clicked", () => {
    window.open = jest.fn();
    render(
      <DocList
        documents={[mockDocuments[3]]}
        settings={mockSettings}
        docsLoading={false}
      />,
    );
    const documentTitle = screen.getByText("Test Document 4");
    userEvent.click(documentTitle);
    expect(window.open).toHaveBeenCalledWith(
      "http://example.com/document.html",
    );
  });

  it("renders the LoadingOverlay when docsLoading is true", () => {
    render(
      <DocList documents={[]} settings={mockSettings} docsLoading={true} />,
    );
    expect(screen.getByRole("presentation")).toBeInTheDocument();
  });

  it("renders a list of DocumentItem components when documents are available", () => {
    render(
      <DocList
        documents={mockDocuments}
        settings={mockSettings}
        docsLoading={false}
      />,
    );
    expect(screen.getAllByText(/Test Document/i)).toHaveLength(
      mockDocuments.length,
    );
  });
});
