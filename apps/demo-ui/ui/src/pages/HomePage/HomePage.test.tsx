// HeaderComponent.test.jsx

import HomePage, { theme } from ".";
import { Settings } from "../../Types";
import { customRender } from "../../test-utils/render";
import { MantineProvider } from "@mantine/core";
import { render as rtlRender } from "@testing-library/react";

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
// jest.mock("./index", () => ({
//   default: () => <div>Mocked HeaderComponent</div>,
// }));
describe("HomePage", () => {
  test("renders logo and settings on larger screens", () => {
    mockUseMediaQuery.mockReturnValue(false);
    const settings = new Settings({
      openSearchIndex: "myIndex",
      modelName: "myModel",
      modelId: "myModelId",
    });
    rtlRender(
      <MantineProvider theme={theme}>
        <HomePage />
      </MantineProvider>,
    );
    // customRender(<HomePage />);
  });
});
