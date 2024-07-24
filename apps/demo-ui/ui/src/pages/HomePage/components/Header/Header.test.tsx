import { HeaderComponent } from ".";
import { render, screen } from "../../../../test-utils";
import { Settings } from "../../../../Types";

const mockUseMediaQuery = jest.fn();
const mockUseMantineTheme = jest.fn();

jest.mock("@mantine/hooks", () => ({
  useMediaQuery: () => mockUseMediaQuery(),
}));
jest.mock("@mantine/core", () => ({
  useMantineTheme: () => mockUseMantineTheme(),
  createStyles: () => () => ({}),
}));
jest.mock("@mantine/prism", () => ({
  Prism: () => <div>Mocked Prism Component</div>,
}));
jest.mock("@mantine/dates", () => ({
  Dates: () => <div>Mocked Date Component</div>,
}));

jest.mock("@emotion/react", () => ({
  ...jest.requireActual("@emotion/react"),
  useTheme: () => ({}),
}));

describe("HeaderComponent", () => {
  it("renders logo and settings on larger screens", () => {
    mockUseMediaQuery.mockReturnValue(false);
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
