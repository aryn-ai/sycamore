import { Dispatch, SetStateAction, useRef, useState } from "react";
import { Settings, SystemChat } from "../../../../Types";
import { ActionIcon, TextInput, useMantineTheme } from "@mantine/core";
import { useMediaQuery } from "@mantine/hooks";
import { createConversation } from "../../../../utils/OpenSearch";
import { setActiveConversation } from ".";
import { IconMessagePlus } from "@tabler/icons-react";

export const NewConversationInput = ({
  refreshConversations,
  setErrorMessage,
  chatInputRef,
  settings,
  setSettings,
  setChatHistory,
  setNavBarOpened,
  loadActiveConversation,
  navBarOpened,
}: {
  refreshConversations: any;
  setErrorMessage: Dispatch<SetStateAction<string | null>>;
  chatInputRef: any;
  settings: Settings;
  setSettings: Dispatch<SetStateAction<Settings>>;
  setChatHistory: Dispatch<SetStateAction<Array<SystemChat>>>;
  setNavBarOpened: any;
  loadActiveConversation: any;
  navBarOpened: boolean;
}) => {
  const [newConversationName, setNewConversationName] = useState("");
  const [error, setError] = useState(false);
  const newConversationInputRef = useRef(null);
  const theme = useMantineTheme();
  const mobileScreen = useMediaQuery(`(max-width: ${theme.breakpoints.sm})`);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(false);
    setNewConversationName(e.target.value);
  };
  async function handleSubmit() {
    try {
      if (newConversationName === "") {
        setError(true);
        return;
      }
      setError(false);
      const createConversationResponse =
        await createConversation(newConversationName);
      const conversationId = createConversationResponse.memory_id;
      setActiveConversation(
        conversationId,
        settings,
        setSettings,
        loadActiveConversation,
      );
      setNavBarOpened(false);
      refreshConversations();
    } catch (e) {
      console.log("Error creating conversation: ", e);
      if (typeof e === "string") {
        setErrorMessage(e.toUpperCase());
      } else if (e instanceof Error) {
        setErrorMessage(e.message);
      }
    } finally {
      setNewConversationName("");
    }
  }
  const handleInputKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleSubmit();
    }
  };
  const handleBlur = () => {
    if (error) {
      setError(false);
    }
  };

  return (
    <>
      <TextInput
        onKeyDown={handleInputKeyPress}
        onChange={handleInputChange}
        ref={newConversationInputRef}
        value={newConversationName}
        w="100%"
        radius="sm"
        fz="xs"
        size={mobileScreen ? "md" : "sm"}
        rightSection={
          mobileScreen ? (
            <ActionIcon size={32} radius="sm" c={error ? "red" : "#5688b0"}>
              <IconMessagePlus size="1rem" stroke={2} onClick={handleSubmit} />
            </ActionIcon>
          ) : (
            ""
          )
        }
        placeholder="New conversation"
        error={error ? "Conversation name cannot be empty" : ""}
        onBlur={handleBlur}
      />
    </>
  );
};
