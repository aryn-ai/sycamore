import { Dispatch, SetStateAction } from "react";
import { Settings, SystemChat } from "../../../../Types";
import { deleteConversation } from "../../../../utils/OpenSearch";
import {
  ActionIcon,
  Anchor,
  Group,
  Loader,
  Menu,
  Text,
  createStyles,
  rem,
} from "@mantine/core";
import { IconDotsVertical, IconMessage, IconTrash } from "@tabler/icons-react";

const useStyles = createStyles((theme) => ({
  link: {
    boxSizing: "border-box",
    display: "inline-block",
    textDecoration: "none",
    borderRadius: theme.radius.md,
    color:
      theme.colorScheme === "dark"
        ? theme.colors.dark[0]
        : theme.colors.gray[7],
    fontSize: theme.fontSizes.sm,
    fontWeight: 500,
    padding: rem(5),
    overflow: "hidden",
    textOverflow: "ellipsis",

    [theme.fn.smallerThan("sm")]: {
      fontSize: theme.fontSizes.lg,
      paddingLeft: rem(15),
      borderRadius: theme.radius.xl,
    },

    "&:hover": {
      backgroundColor:
        theme.colorScheme === "dark"
          ? theme.colors.dark[5]
          : theme.colors.gray[2],
      color: theme.colorScheme === "dark" ? theme.white : theme.black,
    },
  },
  linkRow: {
    width: "100%",
    marginTop: rem(10),
    marginBottom: rem(10),
  },

  linkActive: {
    "&, &:hover": {
      borderLeftColor: theme.fn.variant({
        variant: "filled",
        color: theme.primaryColor,
      }).background,
      backgroundColor: "#5688b0",
      color: theme.white,
    },
  },
  conversationName: {
    [theme.fn.smallerThan("sm")]: {
      width: "55vw",
    },
    [theme.fn.largerThan("sm")]: {
      width: "calc(175px - 7rem)",
    },
    [theme.fn.largerThan("lg")]: {
      width: "calc(275px - 7rem)",
    },
  },
}));

export const NavBarConversationItem = ({
  conversation,
  conversations,
  setConversations,
  selectConversation,
  loading,
  settings,
  setSettings,
  setChatHistory,
  setNavBarOpened,
  openErrorDialog,
}: {
  conversation: any;
  conversations: any[];
  setConversations: any;
  selectConversation: any;
  loading: any;
  settings: any;
  setChatHistory: any;
  setSettings: Dispatch<SetStateAction<Settings>>;
  setNavBarOpened: any;
  openErrorDialog: any;
}) => {
  const { classes, cx } = useStyles();

  const handleDelete = (event: React.MouseEvent) => {
    try {
      event.stopPropagation();
      console.log("Removing ", conversation.id);
      deleteConversation(conversation.id);
      if (conversation.id === settings.activeConversation) {
        setChatHistory(new Array<SystemChat>());
        settings.activeConversation = "";
        setSettings(new Settings(settings));
      }
      const newConversations = conversations.filter(
        (c) => c.id !== conversation.id,
      );

      console.log("newLinks", newConversations);
      setConversations(newConversations);
    } catch (error: any) {
      openErrorDialog("Error deleting conversation: " + error.message);
      console.error("Error deleting conversation: " + error.message);
    }
  };
  return (
    <>
      <Group
        key={conversation.id + "_navbar_row"}
        id={conversation.id + "_navbar_row"}
        noWrap
        className={classes.linkRow}
      >
        <Anchor
          className={cx(classes.link, {
            [classes.linkActive]:
              conversation.id === settings.activeConversation,
          })}
          key={conversation.id}
          onClick={(event) => {
            event.preventDefault();
            selectConversation(conversation.id);
            setNavBarOpened(false);
          }}
          w="100%"
          data-testid="anchorLink"
        >
          <Group position="apart" align="center" noWrap>
            <Group position="left" noWrap pl="xs">
              <IconMessage data-testid="IconMessage" />
              <Text className={classes.conversationName} truncate>
                {conversation.name}
              </Text>
            </Group>
            <Menu shadow="md" position="bottom">
              <Menu.Target>
                <ActionIcon
                  c="white"
                  onClick={(e) => e.stopPropagation()}
                  sx={(theme) => ({
                    borderRadius: 100,
                    "&:hover": {
                      backgroundColor:
                        conversation.id === settings.activeConversation
                          ? "#4f779b"
                          : theme.colors.gray[4],
                    },
                  })}
                  data-testid="three-dots-button"
                >
                  <IconDotsVertical
                    color={
                      conversation.id === settings.activeConversation
                        ? "white"
                        : "gray"
                    }
                  />
                </ActionIcon>
              </Menu.Target>
              <Menu.Dropdown p={0} m={0}>
                <Menu.Item
                  icon={<IconTrash size={14} />}
                  onClick={handleDelete}
                  data-testid="delete-button"
                >
                  Delete
                </Menu.Item>
              </Menu.Dropdown>
            </Menu>
          </Group>

          {loading ? <Loader size="xs" variant="dots" /> : ""}
        </Anchor>
      </Group>
    </>
  );
};
