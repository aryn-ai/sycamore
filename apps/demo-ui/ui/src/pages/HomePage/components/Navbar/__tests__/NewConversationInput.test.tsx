import { render, screen, waitFor, userEvent } from "../../../../../test-utils";
import { useMantineTheme } from "@mantine/core";
import { useMediaQuery } from "@mantine/hooks";
import { NewConversationInput } from "../NewConversationInput";
import { Settings, SystemChat } from "../../../../../Types";
import { createConversation } from "../../../../../utils/OpenSearch";
import "@testing-library/jest-dom";
import { setActiveConversation } from "..";

jest.mock("@mantine/hooks", () => ({
  ...jest.requireActual("@mantine/hooks"),
  useMediaQuery: jest.fn(),
}));

jest.mock("../../../../../utils/OpenSearch", () => ({
  createConversation: jest.fn(),
}));

jest.mock("..", () => ({
  setActiveConversation: jest.fn(),
}));

describe("NewConversationInput", () => {
  beforeEach(() => {
    (useMediaQuery as jest.Mock).mockReturnValue(true);
  });

  it("renders the input field", () => {
    render(
      <NewConversationInput
        refreshConversations={jest.fn()}
        setErrorMessage={jest.fn()}
        chatInputRef={jest.fn()}
        settings={new Settings({})}
        setSettings={jest.fn()}
        setChatHistory={jest.fn()}
        setNavBarOpened={jest.fn()}
        loadActiveConversation={jest.fn()}
        navBarOpened={false}
      />,
    );

    const inputField = screen.getByPlaceholderText(/new conversation/i);
    expect(inputField).toBeInTheDocument();
  });

  it("displays an error when submitting an empty conversation name", async () => {
    render(
      <NewConversationInput
        refreshConversations={jest.fn()}
        setErrorMessage={jest.fn()}
        chatInputRef={jest.fn()}
        settings={new Settings({})}
        setSettings={jest.fn()}
        setChatHistory={jest.fn()}
        setNavBarOpened={jest.fn()}
        loadActiveConversation={jest.fn()}
        navBarOpened={false}
      />,
    );

    const submitButton = screen.getByRole("button");
    userEvent.click(submitButton);

    expect(createConversation).toHaveBeenCalledTimes(0);
  });

  it("creates a new conversation on valid input", async () => {
    (createConversation as jest.Mock).mockResolvedValueOnce({
      memory_id: "new-conversation-id",
    });
    const setNavBarOpenedMock = jest.fn();
    const refreshConversationsMock = jest.fn();

    render(
      <NewConversationInput
        refreshConversations={refreshConversationsMock}
        setErrorMessage={jest.fn()}
        chatInputRef={jest.fn()}
        settings={new Settings({})}
        setSettings={jest.fn()}
        setChatHistory={jest.fn()}
        setNavBarOpened={setNavBarOpenedMock}
        loadActiveConversation={jest.fn()}
        navBarOpened={false}
      />,
    );

    const inputField = screen.getByPlaceholderText(/new conversation/i);
    userEvent.type(inputField, "My New Conversation");

    const submitButton = screen.getByRole("button");
    userEvent.click(submitButton);
    await waitFor(() => {
      expect(createConversation).toHaveBeenCalledWith("My New Conversation");
    });
    expect(setActiveConversation).toHaveBeenCalledWith(
      "new-conversation-id",
      expect.any(Settings),
      expect.any(Function),
      expect.any(Function),
    );

    expect(setNavBarOpenedMock).toHaveBeenCalledWith(false);
    expect(refreshConversationsMock).toHaveBeenCalledTimes(1);
  });

  it("creates a new conversation on Enter key press", async () => {
    (createConversation as jest.Mock).mockResolvedValueOnce({
      memory_id: "new-conversation-id",
    });
    const setNavBarOpenedMock = jest.fn();
    const refreshConversationsMock = jest.fn();

    render(
      <NewConversationInput
        refreshConversations={refreshConversationsMock}
        setErrorMessage={jest.fn()}
        chatInputRef={jest.fn()}
        settings={new Settings({})}
        setSettings={jest.fn()}
        setChatHistory={jest.fn()}
        setNavBarOpened={setNavBarOpenedMock}
        loadActiveConversation={jest.fn()}
        navBarOpened={false}
      />,
    );

    const inputField = screen.getByPlaceholderText(/new conversation/i);
    userEvent.type(inputField, "My New Conversation");
    userEvent.keyboard("{Enter}");

    await waitFor(() => {
      expect(createConversation).toHaveBeenCalledWith("My New Conversation");
    });

    expect(setActiveConversation).toHaveBeenCalledWith(
      "new-conversation-id",
      expect.any(Settings),
      expect.any(Function),
      expect.any(Function),
    );

    expect(setNavBarOpenedMock).toHaveBeenCalledWith(false);
    expect(refreshConversationsMock).toHaveBeenCalledTimes(1);
  });

  it("handles errors during conversation creation", async () => {
    const errorMessage = "Error creating conversation";
    (createConversation as jest.Mock).mockRejectedValue(
      new Error("Error creating conversation"),
    );

    const setErrorMessageMock = jest.fn();

    render(
      <NewConversationInput
        refreshConversations={jest.fn()}
        setErrorMessage={setErrorMessageMock}
        chatInputRef={jest.fn()}
        settings={new Settings({})}
        setSettings={jest.fn()}
        setChatHistory={jest.fn()}
        setNavBarOpened={jest.fn()}
        loadActiveConversation={jest.fn()}
        navBarOpened={false}
      />,
    );

    const inputField = screen.getByPlaceholderText(/new conversation/i);
    userEvent.type(inputField, "New Conversation");

    const submitButton = screen.getByRole("button");
    userEvent.click(submitButton);

    await waitFor(() => {
      expect(setErrorMessageMock).toHaveBeenCalledWith(errorMessage);
    });
  });
});
