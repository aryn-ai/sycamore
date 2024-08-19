import React from "react";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  act,
  userEvent,
} from "../../test-utils";
import PdfViewer from ".";
import { useMediaQuery } from "@mantine/hooks";
import "@testing-library/jest-dom";

jest.mock("@mantine/hooks", () => ({
  ...jest.requireActual("@mantine/hooks"),
  useMediaQuery: jest.fn(),
}));

jest.mock("@react-pdf/renderer", () => ({
  Document: () => <div>Mocked Document</div>,
  Image: () => <div>Mocked Image</div>,
  Page: () => <div>Mocked Page</div>,
  PDFViewer: jest.fn(() => null),
  StyleSheet: { create: () => {} },
  Text: () => <div>Mocked Text</div>,
  View: () => <div>Mocked View</div>,
}));

const mockPdfData = {
  id: "test-pdf-id",
  title: "Test Document",
  url: "https://pdfobject.com/pdf/sample.pdf",
  properties: {},
};

const mockBoxesData = {
  1: [
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
  ],
};

describe("PdfViewer", () => {
  beforeAll(() => {
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ test: 100 }),
        blob: () =>
          Promise.resolve(
            new Blob(["PDF content"], { type: "application/pdf" }),
          ),
      }),
    ) as jest.Mock;
    global.URL.createObjectURL = jest.fn(() => "mocked-url");
  });

  afterAll(() => {
    (global.fetch as jest.Mock).mockRestore();
    (global.URL.createObjectURL as jest.Mock).mockRestore();
  });

  beforeEach(() => {
    localStorage.setItem("pdfDocumentMetadata", JSON.stringify(mockPdfData));
  });

  afterEach(() => {
    localStorage.removeItem("pdfDocumentMetadata");
  });

  it("renders the loading state initially", () => {
    render(<PdfViewer />);
    expect(screen.getByRole("presentation")).toBeInTheDocument();
  });

  it("renders the document, header with title and page numbers on larger screens", async () => {
    (useMediaQuery as jest.Mock).mockReturnValue(false);
    render(<PdfViewer />);

    // Wait for the PDF to load
    await waitFor(() =>
      expect(screen.queryByRole("presentation")).not.toBeInTheDocument(),
    );

    // Verify PDF document and pages
    expect(screen.getAllByText("Test Document")).toHaveLength(1);
    expect(screen.getAllByText("Loading PDF…")).toHaveLength(1);
    expect(screen.getByText("1 / -1")).toBeInTheDocument();
  });

  it("renders the header with title and info button on mobile", async () => {
    (useMediaQuery as jest.Mock).mockReturnValue(true); // Simulate mobile screen

    render(<PdfViewer />);

    await waitFor(() =>
      expect(screen.queryByRole("presentation")).not.toBeInTheDocument(),
    );

    expect(screen.getByText("Test Document")).toBeInTheDocument();
    expect(screen.getByTestId("infoButton")).toBeInTheDocument();
  });

  // it("navigates between pages", async () => {
  //   (useMediaQuery as jest.Mock).mockReturnValue(false);

  //   render(<PdfViewer />);
  //   await waitFor(() =>
  //     expect(screen.queryByRole("presentation")).not.toBeInTheDocument(),
  //   );
  //   expect(screen.getByText("Test Document")).toBeInTheDocument();

  //   // Mock the setPageNumber function
  //   const setPageNumberMock = jest.fn();
  //   jest
  //     .spyOn(React, "useState")
  //     .mockImplementation(() => [1, setPageNumberMock]);

  //   const nextButton = await screen.getByTestId("nextButton");
  //   userEvent.click(nextButton);
  //   expect(setPageNumberMock).toHaveBeenCalledWith(2);

  //   const prevButton = screen.getByTestId("prevButton");
  //   userEvent.click(prevButton);
  //   expect(setPageNumberMock).toHaveBeenCalledWith(1);

  //   // Test edge cases
  //   // First page
  //   userEvent.click(prevButton);
  //   expect(setPageNumberMock).toHaveBeenCalledWith(1);

  //   // Last page (assuming 5 pages for this test)
  //   jest
  //     .spyOn(React, "useState")
  //     .mockImplementation(() => [5, setPageNumberMock]);
  //   userEvent.click(nextButton);
  //   expect(setPageNumberMock).toHaveBeenCalledWith(5);
  // });

  // it("zooms in and out", async () => {
  //   render(<PdfViewer />);
  //   await screen.findByText(/Test Document/i);

  //   const zoomInButton = screen.getByTestId("zoomInButton");
  //   const zoomOutButton = screen.getByTestId("zoomOutButton");

  //   // Mock the setScale function
  //   const setScaleMock = jest.fn();
  //   jest.spyOn(React, "useState").mockImplementation(() => [1.5, setScaleMock]);

  //   // Test zoom in
  //   userEvent.click(zoomInButton);
  //   expect(setScaleMock).toHaveBeenCalledWith(1.7);

  //   userEvent.click(zoomInButton);
  //   expect(setScaleMock).toHaveBeenCalledWith(1.9);

  //   // Test zoom out
  //   jest.spyOn(React, "useState").mockImplementation(() => [1.5, setScaleMock]);

  //   userEvent.click(zoomOutButton);
  //   expect(setScaleMock).toHaveBeenCalledWith(1.3);

  //   // Test minimum zoom level
  //   jest.spyOn(React, "useState").mockImplementation(() => [0.2, setScaleMock]);

  //   userEvent.click(zoomOutButton);
  //   expect(setScaleMock).toHaveBeenCalledWith(0.2);
  // });

  // Add tests for other interactions:
  // - Clicking on the info button
  // - Opening external links
  // - Error scenarios (failed PDF fetching)
  // - Box rendering (if applicable)
});
