import { useMediaQuery } from "@mantine/hooks";
import { HeaderComponent } from ".";
import { fireEvent, render, screen, userEvent } from "../../../../test-utils";
import { Settings } from "../../../../Types";
import "@testing-library/jest-dom";

jest.mock("@mantine/hooks", () => ({
  ...jest.requireActual("@mantine/hooks"),
  useMediaQuery: jest.fn(),
}));
const settings = new Settings({
  openSearchIndex: "myIndex",
  modelName: "myModel",
  modelId: "myModelId",
});

describe("HeaderComponent", () => {
  it("renders logo and settings on larger screens", () => {
    (useMediaQuery as jest.Mock).mockReturnValue(false);

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
  test("renders burger menu and logo on smaller screens", () => {
    (useMediaQuery as jest.Mock).mockReturnValue(true);
    render(
      <HeaderComponent
        navBarOpened={false}
        setNavBarOpened={jest.fn()}
        settings={settings}
      />,
    );

    expect(screen.getByRole("img")).toBeInTheDocument();
    expect(screen.getByRole("button")).toBeInTheDocument();

    expect(screen.queryByText(/index: myIndex/i)).toBeNull();
    expect(screen.queryByText(/llm model: myModel/i)).toBeNull();
    expect(screen.queryByText(/llm model id: myModelId/i)).toBeNull();
  });

  test("toggles navbar on burger menu click", () => {
    (useMediaQuery as jest.Mock).mockReturnValue(true);
    const setNavBarOpenedMock = jest.fn();
    render(
      <HeaderComponent
        navBarOpened={false}
        setNavBarOpened={setNavBarOpenedMock}
        settings={settings}
      />,
    );

    const burgerButton = screen.getByRole("button");

    userEvent.click(burgerButton);
    expect(setNavBarOpenedMock).toHaveBeenCalledTimes(1);
    userEvent.click(burgerButton);
    expect(setNavBarOpenedMock).toHaveBeenCalledTimes(2);
  });
});
