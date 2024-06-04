import React, { useRef, useState } from "react";
import { ChatBox } from "./components/Chatbox";
import {
  ConversationListNavbar,
  setActiveConversation,
} from "./components/Navbar";
import {
  Alert,
  AppShell,
  Dialog,
  MantineProvider,
  useMantineTheme,
} from "@mantine/core";
import { SearchResultDocument, Settings, SystemChat } from "../../Types";
import { IconAlertCircle } from "@tabler/icons-react";
import {
  getConversations,
  getFeedback,
  getInteractions,
} from "../../utils/OpenSearch";
import { useDisclosure, useMediaQuery } from "@mantine/hooks";
import { thumbToBool } from "../../utils/ChatboxUtils";
import { HeaderComponent } from "./components/Header";

export default function HomePage() {
  const [settings, setSettings] = useState(new Settings());
  const [searchResults, setSearchResults] = useState(
    new Array<SearchResultDocument>(),
  );
  const [streaming, setStreaming] = useState(false);
  const [docsLoading, setDocsLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState(new Array<SystemChat>());
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [navBarOpened, setNavBarOpened] = useState(false);
  const [loadingConversation, setLoadingConversation] = useState(false);
  const [conversations, setConversations] = useState<any>([]);
  const chatInputRef = useRef<HTMLInputElement | null>(null);
  const theme = useMantineTheme();
  const mobileScreen = useMediaQuery(`(max-width: ${theme.breakpoints.sm})`);
  const [errorDialogMessage, setErrorDialogMessage] = useState("");
  const [errorDialogOpened, errorDialogHandler] = useDisclosure(false);

  const openErrorDialog = (message: string, timeout: number = 3000) => {
    setErrorDialogMessage(message);
    errorDialogHandler.open();
    setTimeout(() => {
      errorDialogHandler.close();
    }, 3000);
  };

  const reset = () => {
    setStreaming(false);
    setDocsLoading(false);
    setSearchResults([]);
    setChatHistory([]);
    setErrorMessage(null);
  };
  const loadActiveConversation = () => {
    const populateConversationMessages = async () => {
      try {
        setLoadingConversation(true);
        setStreaming(true);
        console.log("Loading convos");
        const interactionsResponse = await getInteractions(
          settings.activeConversation,
        );
        let previousInteractions = new Array<SystemChat>();
        let interactionsData = await interactionsResponse;
        if ("hits" in interactionsData) {
          interactionsData = interactionsData.hits.hits;
        } else {
          interactionsData = interactionsData.interactions;
        }
        console.info("interactionsData: ", interactionsData);
        const previousInteractionsUnFlattened = await Promise.all(
          interactionsData.map(async (interaction_raw: any) => {
            const interaction = interaction_raw._source ?? interaction_raw;
            const interaction_id =
              interaction_raw._id ?? interaction_raw.interaction_id;
            const feedback = await getFeedback(interaction_id);
            const systemChat = new SystemChat({
              id: interaction_id + "_response",
              response: interaction.response,
              interaction_id: interaction_id,
              modelName: null,
              queryUsed: interaction.input,
              feedback: feedback.found
                ? thumbToBool(feedback._source.thumb)
                : null,
              comment: feedback.found ? feedback._source.comment : "",
            });
            return systemChat;
          }),
        );
        previousInteractionsUnFlattened.forEach((chat) => {
          previousInteractions = [chat, ...previousInteractions];
        });
        console.log("Setting previous interactions", previousInteractions);
        setChatHistory(previousInteractions);
        setLoadingConversation(false);
        setStreaming(false);
      } catch (error: any) {
        openErrorDialog("Error loading messages: " + error.message);
        console.error("Error loading messages: " + error.message);
      }
    };
    populateConversationMessages();
  };

  async function refreshConversations() {
    try {
      let result: any = [];
      const getConversationsResult = await getConversations();
      const retrievedConversations: { conversations: any } = {
        conversations: null,
      };
      if ("conversations" in getConversationsResult) {
        retrievedConversations.conversations =
          getConversationsResult.conversations;
      } else {
        retrievedConversations.conversations = getConversationsResult.memories;
      }
      retrievedConversations.conversations.forEach((conversation: any) => {
        result = [
          {
            id: conversation.conversation_id ?? conversation.memory_id,
            name: conversation.name,
            created_at: conversation.create_time,
          },
          ...result,
        ];
      });
      setConversations(result);
      if (result.length > 0 && settings.activeConversation == "") {
        setActiveConversation(
          result[0].id,
          settings,
          setSettings,
          loadActiveConversation,
        );
      }
    } catch (error: any) {
      openErrorDialog("Error fetching conversation: " + error.message);
      console.error("Error fetching conversation: " + error.message);
    }
  }

  return (
    <MantineProvider
      theme={{
        fontFamily:
          '"Segoe UI",Roboto,Helvetica Neue,Arial,"Noto Sans",sans-serif,"Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol","Noto Color Emoji"',
        fontFamilyMonospace: "Monaco, Courier, monospace",
        headings: { fontFamily: "Greycliff CF, sans-serif" },
        fontSizes: {
          xs: "0.8rem",
          sm: "1rem",
          md: "1.1rem",
          lg: "1.2rem",
          xl: "1.4rem",
        },
      }}
    >
      <AppShell
        padding="md"
        header={
          <HeaderComponent
            navBarOpened={navBarOpened}
            setNavBarOpened={setNavBarOpened}
            settings={settings}
          />
        }
        navbarOffsetBreakpoint="sm"
        navbar={
          <ConversationListNavbar
            navBarOpened={navBarOpened}
            settings={settings}
            setSettings={setSettings}
            setErrorMessage={setErrorMessage}
            loadingConversation={loadingConversation}
            loadActiveConversation={loadActiveConversation}
            conversations={conversations}
            setConversations={setConversations}
            refreshConversations={refreshConversations}
            setChatHistory={setChatHistory}
            chatInputRef={chatInputRef}
            setNavBarOpened={setNavBarOpened}
            openErrorDialog={openErrorDialog}
          ></ConversationListNavbar>
        }
        styles={() => ({
          main: { backgroundColor: "white", paddingBottom: 0 },
        })}
      >
        <Dialog
          opened={errorDialogOpened}
          size="lg"
          radius="md"
          position={{ right: 10, top: 10 }}
          transition="slide-left"
          transitionDuration={300}
          transitionTimingFunction="ease"
          p="none"
        >
          <Alert
            icon={<IconAlertCircle size="1rem" />}
            title="Error"
            color="red"
            withCloseButton
            onClose={errorDialogHandler.close}
          >
            {errorDialogMessage}
          </Alert>
        </Dialog>
        <ChatBox
          chatHistory={chatHistory}
          searchResults={searchResults}
          setChatHistory={setChatHistory}
          setSearchResults={setSearchResults}
          streaming={streaming}
          setStreaming={setStreaming}
          setDocsLoading={setDocsLoading}
          setErrorMessage={setErrorMessage}
          settings={settings}
          setSettings={setSettings}
          refreshConversations={refreshConversations}
          chatInputRef={chatInputRef}
          openErrorDialog={openErrorDialog}
        />
      </AppShell>
    </MantineProvider>
  );
}
