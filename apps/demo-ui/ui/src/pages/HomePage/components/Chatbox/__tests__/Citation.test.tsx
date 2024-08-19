import { render, screen, userEvent } from "../../../../../test-utils";
import { Citation } from "../Citation";
import { SearchResultDocument } from "../../../../../Types";
import { useMantineTheme } from "@mantine/core";
import "@testing-library/jest-dom";

jest.mock("@mantine/core", () => ({
  ...jest.requireActual("@mantine/core"),
  useMantineTheme: jest.fn(),
}));

describe("Citation", () => {
  beforeAll(() => {
    (useMantineTheme as jest.Mock).mockReturnValue({
      colors: {
        blue: [
          "#not-used",
          "#not-used",
          "#not-used",
          "#not-used",
          "#not-used",
          "#007bff",
        ],
        gray: [
          "#not-used",
          "#not-used",
          "#not-used",
          "#not-used",
          "#not-used",
          "#6c757d",
        ],
      },
    });
    Object.defineProperty(window, "localStorage", {
      value: {
        setItem: jest.fn(),
        getItem: jest.fn(),
      },
      writable: true,
    });

    window.open = jest.fn();
  });
  const linkProps = {
    document: new SearchResultDocument({
      id: "1",
      url: "http://example.com",
      title: "Example Document",
      index: 0,
      description: "Example description",
      relevanceScore: "0",
      properties: {},
      isPdf: jest.fn().mockReturnValue(false),
    }),
    citationNumber: 1,
  };
  const pdfProps = {
    document: new SearchResultDocument({
      id: "1",
      url: "http://example.com",
      title: "Example Document",
      index: 0,
      description: "Example description",
      relevanceScore: "0",
      properties: {},
      isPdf: jest.fn().mockReturnValue(true),
    }),
    citationNumber: 1,
  };
  const htmlProps = {
    document: new SearchResultDocument({
      id: "1",
      url: "http://example.com/document.html",
      title: "Example Document",
      index: 0,
      description: "Example description",
      relevanceScore: "0",
      properties: {},
      isPdf: jest.fn().mockReturnValue(false),
    }),
    citationNumber: 1,
  };

  it("renders the citation badge with the correct number", () => {
    render(<Citation {...linkProps} />);
    expect(screen.getByText("1")).toBeInTheDocument();
  });

  it("renders the document title and URL in the HoverCardDropdown", () => {
    render(<Citation {...linkProps} />);
    userEvent.hover(screen.getByText("1"));
    expect(screen.getByText("Example Document")).toBeInTheDocument();
    expect(screen.getByText("http://example.com")).toBeInTheDocument();
  });

  it("renders the correct icon based on document type (Link)", () => {
    render(<Citation {...linkProps} />);
    userEvent.hover(screen.getByText("1"));
    expect(screen.getByTestId("icon-link")).toBeInTheDocument();
  });

  it("renders the correct icon based on document type (PDF)", () => {
    render(<Citation {...pdfProps} />);
    userEvent.hover(screen.getByText("1"));
    expect(screen.getByTestId("icon-pdf")).toBeInTheDocument();
  });

  it("renders the correct icon based on document type (HTML)", () => {
    render(<Citation {...htmlProps} />);
    userEvent.hover(screen.getByText("1"));
    expect(screen.getByTestId("icon-html")).toBeInTheDocument();
  });

  it("opens the PDF viewer when a PDF citation is clicked", () => {
    render(<Citation {...pdfProps} />);
    const anchor = screen.getByText("1");
    userEvent.click(anchor);
    expect(localStorage.setItem).toHaveBeenCalledWith(
      "pdfDocumentMetadata",
      JSON.stringify(pdfProps.document),
    );
    expect(window.open).toHaveBeenCalledWith("/viewPdf");
  });

  it("opens the document URL in a new tab when a non-PDF citation is clicked", () => {
    render(<Citation {...linkProps} />);
    const anchor = screen.getByText("1");
    userEvent.click(anchor);
    expect(window.open).toHaveBeenCalledWith("http://example.com");
  });
});
