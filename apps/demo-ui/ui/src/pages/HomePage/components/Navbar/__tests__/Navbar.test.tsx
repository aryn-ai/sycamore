import React from "react";
import { render, screen, waitFor } from "../../../../../test-utils";
import { Settings, SystemChat } from "../../../../../Types";
import userEvent from "@testing-library/user-event";
import { createConversation } from "../../../../../utils/OpenSearch";
import { ConversationListNavbar, setActiveConversation } from "..";
import "@testing-library/jest-dom";

jest.mock("../../../../../utils/OpenSearch", () => ({
  createConversation: jest.fn(),
}));

const mockConversations = [
  { id: "123", name: "Conversation 123" },
  { id: "456", name: "Conversation 456" },
];

const mockSettings = new Settings({
  activeConversation: "123",
});

describe("ConversationListNavbar", () => {
  it("renders loader while loading", () => {
    render(
      <ConversationListNavbar
        navBarOpened={true}
        settings={mockSettings}
        setSettings={jest.fn()}
        setErrorMessage={jest.fn()}
        loadingConversation={true}
        loadActiveConversation={jest.fn()}
        conversations={[]}
        refreshConversations={jest.fn()}
        setConversations={jest.fn()}
        setChatHistory={jest.fn()}
        chatInputRef={React.createRef()}
        setNavBarOpened={jest.fn()}
        openErrorDialog={jest.fn()}
      />,
    );
    expect(screen.getByRole("presentation")).toBeInTheDocument();
  });

  it("renders conversations list and input component", async () => {
    render(
      <ConversationListNavbar
        navBarOpened={true}
        settings={mockSettings}
        setSettings={jest.fn()}
        setErrorMessage={jest.fn()}
        loadingConversation={false}
        loadActiveConversation={jest.fn()}
        conversations={mockConversations}
        refreshConversations={jest.fn()}
        setConversations={jest.fn()}
        setChatHistory={jest.fn()}
        chatInputRef={React.createRef()}
        setNavBarOpened={jest.fn()}
        openErrorDialog={jest.fn()}
      />,
    );

    expect(screen.getAllByText(/Conversation/i)).toHaveLength(2);

    const inputField = screen.getByPlaceholderText(/new conversation/i);
    expect(inputField).toBeInTheDocument();
  });

  it("creates a new conversation on valid input", async () => {
    (createConversation as jest.Mock).mockResolvedValueOnce({
      memory_id: "new-conversation-id",
    });

    render(
      <ConversationListNavbar
        navBarOpened={true}
        settings={mockSettings}
        setSettings={jest.fn()}
        setErrorMessage={jest.fn()}
        loadingConversation={false}
        loadActiveConversation={jest.fn()}
        conversations={mockConversations}
        refreshConversations={jest.fn()}
        setConversations={jest.fn()}
        setChatHistory={jest.fn()}
        chatInputRef={React.createRef()}
        setNavBarOpened={jest.fn()}
        openErrorDialog={jest.fn()}
      />,
    );

    const inputField = screen.getByPlaceholderText(/new conversation/i);
    userEvent.type(inputField, "My New Conversation");

    userEvent.keyboard("{Enter}");

    await waitFor(() => {
      expect(createConversation).toHaveBeenCalledWith("My New Conversation");
    });
  });

  it("handles errors during conversation creation", async () => {
    const errorMessage = "Error creating conversation";
    (createConversation as jest.Mock).mockRejectedValue(
      new Error("Error creating conversation"),
    );

    const openErrorDialogMock = jest.fn();
    const setErrorMessageMock = jest.fn();

    render(
      <ConversationListNavbar
        navBarOpened={true}
        settings={mockSettings}
        setSettings={jest.fn()}
        setErrorMessage={setErrorMessageMock}
        loadingConversation={false}
        loadActiveConversation={jest.fn()}
        conversations={mockConversations}
        refreshConversations={jest.fn()}
        setConversations={jest.fn()}
        setChatHistory={jest.fn()}
        chatInputRef={React.createRef()}
        setNavBarOpened={jest.fn()}
        openErrorDialog={openErrorDialogMock}
      />,
    );

    const inputField = screen.getByPlaceholderText(/new conversation/i);
    userEvent.type(inputField, "New Conversation");

    userEvent.keyboard("{Enter}");

    await waitFor(() => {
      expect(setErrorMessageMock).toHaveBeenCalledWith(errorMessage);
    });
  });
});
