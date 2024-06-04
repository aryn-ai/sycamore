import { useState } from "react";
import { Settings, SystemChat } from "../../../../Types";
import { updateFeedback } from "../../../../utils/OpenSearch";
import { ActionIcon, Group, TextInput } from "@mantine/core";
import {
  IconThumbDown,
  IconThumbDownFilled,
  IconThumbUp,
  IconThumbUpFilled,
} from "@tabler/icons-react";

export const FeedbackButtons = ({
  systemChat,
  settings,
}: {
  systemChat: SystemChat;
  settings: Settings;
}) => {
  const [thumbUpState, setThumbUp] = useState(systemChat.feedback);
  const [comment, setComment] = useState(systemChat.comment);
  const handleSubmit = async (thumb: boolean | null) => {
    updateFeedback(
      settings.activeConversation,
      systemChat.interaction_id,
      thumb,
      comment,
    );
  };
  const handleInputKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleSubmit(thumbUpState);
    }
  };
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setComment(e.target.value);
  };
  return (
    <Group position="left" spacing="xs">
      <Group>
        <ActionIcon
          size={32}
          radius="xs"
          component="button"
          onClick={(event) => {
            if (thumbUpState == null || !thumbUpState) {
              setThumbUp(true);
              systemChat.feedback = true;
              handleSubmit(true);
            } else {
              setThumbUp(null);
              systemChat.feedback = null;
              handleSubmit(null);
            }
          }}
        >
          {thumbUpState == null || !thumbUpState ? (
            <IconThumbUp size="1rem" />
          ) : (
            <IconThumbUpFilled size="1rem" color="green" fill="green" />
          )}
        </ActionIcon>
        <ActionIcon
          size={32}
          radius="xs"
          component="button"
          onClick={(event) => {
            if (thumbUpState == null || thumbUpState) {
              setThumbUp(false);
              systemChat.feedback = false;
              handleSubmit(false);
            } else {
              setThumbUp(null);
              systemChat.feedback = null;
              handleSubmit(null);
            }
          }}
        >
          {thumbUpState == null || thumbUpState ? (
            <IconThumbDown size="1rem" />
          ) : (
            <IconThumbDownFilled size="1rem" color="red" fill="red" />
          )}
        </ActionIcon>
      </Group>
      <TextInput
        onKeyDown={handleInputKeyPress}
        onChange={handleInputChange}
        value={comment}
        radius="sm"
        fz="xs"
        fs="italic"
        color="blue"
        variant="unstyled"
        placeholder="Leave a comment"
      />
    </Group>
  );
};
