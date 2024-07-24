import { HeaderComponent } from ".";
import { render, screen } from "../../../../test-utils";
import { Settings } from "../../../../Types";
import "@testing-library/jest-dom";

describe("HeaderComponent", () => {
  it("renders logo and settings on larger screens", () => {
    const settings = new Settings({
      openSearchIndex: "myIndex",
      modelName: "myModel",
      modelId: "myModelId",
    });

    render(
      <HeaderComponent
        navBarOpened={false}
        setNavBarOpened={jest.fn()}
        settings={settings}
      />,
    );

    expect(screen.getByRole("img")).toBeInTheDocument();
    expect(screen.getByText(/index: myIndex/i)).toBeInTheDocument();
    expect(screen.getByText(/llm model: myModel/i)).toBeInTheDocument();
    expect(screen.getByText(/llm model id: myModelId/i)).toBeInTheDocument();
  });
});
