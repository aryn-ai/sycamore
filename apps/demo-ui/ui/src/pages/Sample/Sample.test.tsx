import { render, screen } from "@testing-library/react";
import Sample from "./Sample";
import "@testing-library/jest-dom";

describe("Sample component", () => {
  it("renders message", () => {
    render(<Sample />);

    const headingElement = screen.getByRole("heading", {
      name: /Hello, Soham!/i,
    });

    expect(headingElement).toBeInTheDocument();
  });
});
