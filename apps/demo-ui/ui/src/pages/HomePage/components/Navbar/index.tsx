import React, { Dispatch, SetStateAction, useState, useEffect } from "react";
import {
  createStyles,
  Loader,
  Navbar,
  useMantineTheme,
  rem,
  Center,
  Container,
} from "@mantine/core";
import { Settings } from "../../../../Types";
import { NavBarConversationItem } from "./NavBarConversationItem";
import { NewConversationInput } from "./NewConversationInput";

const useStyles = createStyles((theme) => ({
  wrapper: {
    display: "flex",
    padding: "1rem",
  },
  main: {
    flex: 1,
    width: "100%",
    backgroundColor:
      theme.colorScheme === "dark"
        ? theme.colors.dark[6]
        : theme.colors.gray[0],
  },
  title: {
    textTransform: "uppercase",
    letterSpacing: rem(-0.25),
  },
}));

export function setActiveConversation(
  conversationId: string,
  settings: Settings,
  setSettings: Dispatch<SetStateAction<Settings>>,
  loadActiveConversation: any,
) {
  settings.activeConversation = conversationId;
  setSettings(settings);
  loadActiveConversation(conversationId);
}

export const ConversationListNavbar = ({
  navBarOpened,
  settings,
  setSettings,
  setErrorMessage,
  loadingConversation,
  loadActiveConversation,
  conversations,
  refreshConversations,
  setConversations,
  setChatHistory,
  chatInputRef,
  setNavBarOpened,
  openErrorDialog,
}: {
  navBarOpened: boolean;
  settings: Settings;
  setSettings: Dispatch<SetStateAction<Settings>>;
  setErrorMessage: Dispatch<SetStateAction<string | null>>;
  loadingConversation: boolean;
  loadActiveConversation: any;
  conversations: any;
  refreshConversations: any;
  setConversations: any;
  setChatHistory: any;
  chatInputRef: any;
  setNavBarOpened: any;
  openErrorDialog: any;
}) => {
  const theme = useMantineTheme();
  const { classes, cx } = useStyles();
  const [loading, setLoading] = useState(false);

  const selectConversation = (conversationId: string) => {
    console.info("Set active conversation to ", conversationId);
    setActiveConversation(
      conversationId,
      settings,
      setSettings,
      loadActiveConversation,
    );
    refreshConversations();
  };

  useEffect(() => {
    setLoading(true);
    refreshConversations();
    setLoading(false);
  }, []);
  return (
    <Navbar
      hiddenBreakpoint="sm"
      height="100%"
      hidden={!navBarOpened}
      width={{ sm: 200, lg: 300 }}
      sx={{
        overflow: "hidden",
        transition: "width 150ms ease, min-width 150ms ease",
        backgroundColor:
          theme.colorScheme === "dark"
            ? theme.colors.dark[6]
            : theme.colors.gray[0],
      }}
    >
      {loading ? (
        <Center p="md">
          <Loader size="xs" />
        </Center>
      ) : null}
      <Navbar.Section
        p="xs"
        m="xs"
        sx={{ display: "flex", alignItems: "center", justifyContent: "center" }}
      >
        <NewConversationInput
          refreshConversations={refreshConversations}
          setErrorMessage={setErrorMessage}
          chatInputRef={chatInputRef}
          settings={settings}
          setSettings={setSettings}
          setChatHistory={setChatHistory}
          setNavBarOpened={setNavBarOpened}
          loadActiveConversation={loadActiveConversation}
          navBarOpened={navBarOpened}
        />
      </Navbar.Section>
      <Navbar.Section grow className={classes.wrapper} w={{ sm: 200, lg: 300 }}>
        {loadingConversation ? (
          <Container m="0">
            <Center p="md">
              <Loader size="sm" variant="dots" />
            </Center>
          </Container>
        ) : (
          <div className={classes.main}>
            {conversations.map((conversation: any) => (
              <NavBarConversationItem
                key={conversation.id}
                conversation={conversation}
                conversations={conversations}
                setConversations={setConversations}
                selectConversation={selectConversation}
                loading={loading}
                settings={settings}
                setSettings={setSettings}
                setChatHistory={setChatHistory}
                setNavBarOpened={setNavBarOpened}
                openErrorDialog={openErrorDialog}
              />
            ))}
          </div>
        )}
      </Navbar.Section>
    </Navbar>
  );
};
