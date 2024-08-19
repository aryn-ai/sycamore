import { render, screen, fireEvent, waitFor } from "../../../../../test-utils";
import userEvent from "@testing-library/user-event";
import { NavBarConversationItem } from "../NavBarConversationItem";
import { Settings, SystemChat } from "../../../../../Types";
import { deleteConversation } from "../../../../../utils/OpenSearch";
import "@testing-library/jest-dom";

jest.mock("../../../../../utils/OpenSearch", () => ({
  deleteConversation: jest.fn(),
}));

const mockConversation = {
  id: "123",
  name: "Test Conversation",
};

const mockConversations = [
  mockConversation,
  { id: "456", name: "Another Conversation" },
];

describe("NavBarConversationItem", () => {
  it("renders conversation name and icon", () => {
    render(
      <NavBarConversationItem
        conversation={mockConversation}
        conversations={mockConversations}
        setConversations={jest.fn()}
        selectConversation={jest.fn()}
        loading={false}
        settings={new Settings({ activeConversation: "123" })}
        setSettings={jest.fn()}
        setChatHistory={jest.fn()}
        setNavBarOpened={jest.fn()}
        openErrorDialog={jest.fn()}
      />,
    );

    expect(screen.getByText("Test Conversation")).toBeInTheDocument();
    expect(screen.getByTestId("IconMessage")).toBeInTheDocument();
  });

  it("highlights the active conversation", () => {
    render(
      <NavBarConversationItem
        conversation={mockConversation}
        conversations={mockConversations}
        setConversations={jest.fn()}
        selectConversation={jest.fn()}
        loading={false}
        settings={{ activeConversation: "123" }}
        setSettings={jest.fn()}
        setChatHistory={jest.fn()}
        setNavBarOpened={jest.fn()}
        openErrorDialog={jest.fn()}
      />,
    );

    const anchor = screen.getByTestId("anchorLink");
    expect(anchor).toHaveStyle({ color: "#FFFFFF" });
  });

  it("does not highlight an inactive conversation", () => {
    render(
      <NavBarConversationItem
        conversation={mockConversation}
        conversations={mockConversations}
        setConversations={jest.fn()}
        selectConversation={jest.fn()}
        loading={false}
        settings={{ activeConversation: "1234" }}
        setSettings={jest.fn()}
        setChatHistory={jest.fn()}
        setNavBarOpened={jest.fn()}
        openErrorDialog={jest.fn()}
      />,
    );

    const anchor = screen.getByTestId("anchorLink");
    expect(anchor).toHaveStyle({ color: "#000000" });
  });

  it("calls selectConversation on click", async () => {
    const selectConversationMock = jest.fn();
    const setNavBarOpenedMock = jest.fn();

    render(
      <NavBarConversationItem
        conversation={mockConversation}
        conversations={mockConversations}
        setConversations={jest.fn()}
        selectConversation={selectConversationMock}
        loading={false}
        settings={{ activeConversation: "" }}
        setSettings={jest.fn()}
        setChatHistory={jest.fn()}
        setNavBarOpened={setNavBarOpenedMock}
        openErrorDialog={jest.fn()}
      />,
    );

    const anchor = screen.getByTestId("anchorLink");
    await userEvent.click(anchor);

    expect(selectConversationMock).toHaveBeenCalledWith("123");
    expect(setNavBarOpenedMock).toHaveBeenCalledWith(false);
  });

  it("deletes conversation and updates state on delete", async () => {
    const setConversationsMock = jest.fn();
    const setSettingsMock = jest.fn();
    const setChatHistoryMock = jest.fn();
    render(
      <NavBarConversationItem
        conversation={mockConversation}
        conversations={mockConversations}
        setConversations={setConversationsMock}
        selectConversation={jest.fn()}
        loading={false}
        settings={{ activeConversation: "123" }}
        setSettings={setSettingsMock}
        setChatHistory={setChatHistoryMock}
        setNavBarOpened={jest.fn()}
        openErrorDialog={jest.fn()}
      />,
    );
    const threeDotsButton = screen.getByTestId("three-dots-button");
    await userEvent.click(threeDotsButton);

    const deleteButton = screen.getByTestId("delete-button");
    await userEvent.click(deleteButton);

    await waitFor(() => {
      expect(deleteConversation).toHaveBeenCalledWith("123");
      expect(setChatHistoryMock).toHaveBeenCalledWith(new Array<SystemChat>());
      expect(setSettingsMock).toHaveBeenCalledWith(
        expect.objectContaining({
          activeConversation: "",
        }),
      );
      expect(setConversationsMock).toHaveBeenCalledWith([mockConversations[1]]);
    });
  });
});
