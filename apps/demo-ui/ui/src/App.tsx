import React, { useRef, useState } from 'react';
import { ChatBox, thumbToBool } from './Chatbox'
import { ControlPanel } from './Controlpanel'
import { ConversationListNavbar, setActiveConversation } from './ConversationList'
import { DocList } from './Doclist'
import { AppShell, Burger, Container, Footer, Grid, Group, Header, Image, MantineProvider, MediaQuery, Notification, Stack, Text, useMantineTheme } from '@mantine/core';
import { SearchResultDocument, Settings, SystemChat, UserChat } from './Types';
import { IconX } from '@tabler/icons-react';
import { getConversations, getFeedback, getInteractions } from './OpenSearch';
import { useMediaQuery } from '@mantine/hooks';


export default function App() {

  const [settings, setSettings] = useState(new Settings());
  const [searchResults, setSearchResults] = useState(new Array<SearchResultDocument>());
  const [streaming, setStreaming] = useState(false);
  const [docsLoading, setDocsLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState(new Array<SystemChat>())
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [navBarOpened, setNavBarOpened] = useState(false);
  const [loadingConversation, setLoadingConversation] = useState(false);
  const [conversations, setConversations] = useState<any>([]);
  const chatInputRef = useRef<HTMLInputElement | null>(null);
  const theme = useMantineTheme();
  const mobileScreen = useMediaQuery(`(max-width: ${theme.breakpoints.sm})`);


  const reset = () => {
    setStreaming(false);
    setDocsLoading(false);
    setSearchResults([]);
    setChatHistory([]);
    setErrorMessage(null);
  }

  const ErrorNotification = ({ message }: { message: string | null }) => {
    return (
      <Notification sx={{ "position": "absolute", "z-index": "5" }} maw="30rem" icon={<IconX size="1.1rem" />} color="red" onClose={() => setErrorMessage(null)} >
        {message}
      </Notification>
    )
  }
  const loadActiveConversation = () => {
    const populateConversationMessages = async () => {
      setLoadingConversation(true)
      setStreaming(true)
      console.log("Loading convos")
      const interactionsResponse = await getInteractions(settings.activeConversation)
      var previousInteractions = new Array<SystemChat>()
      let interactionsData = await interactionsResponse;
      if ("hits" in interactionsData) {
        interactionsData = interactionsData.hits.hits
      } else {
        interactionsData = interactionsData.interactions
      }
      console.info("interactionsData: ", interactionsData)
      const previousInteractionsUnFlattened = await Promise.all(interactionsData.map(async (interaction_raw: any) => {
        const interaction = interaction_raw._source ?? interaction_raw
        const interaction_id = interaction_raw._id ?? interaction_raw.interaction_id
        const feedback = await getFeedback(interaction_id)
        const systemChat = new SystemChat(
          {
            id: interaction_id + "_response",
            response: interaction.response,
            interaction_id: interaction_id,
            modelName: null,
            queryUsed: interaction.input,
            feedback: feedback.found ? thumbToBool(feedback._source.thumb) : null,
            comment: feedback.found ? feedback._source.comment : ""
          });
        return systemChat;
      }));
      previousInteractionsUnFlattened.forEach((chat) => {
        previousInteractions = [chat, ...previousInteractions]
      })
      console.log("Setting previous interactions", previousInteractions)
      setChatHistory(previousInteractions.reverse())
      setLoadingConversation(false)
      setStreaming(false)
    }
    populateConversationMessages();
  }

  async function refreshConversations() {
    let result: any = []
    const getConversationsResult = await getConversations();
    let retrievedConversations: { conversations: any } = { conversations: null };
    if ("conversations" in getConversationsResult) {
        retrievedConversations.conversations = getConversationsResult.conversations;
    } else {
        retrievedConversations.conversations = getConversationsResult.memories;
    }
    retrievedConversations.conversations.forEach((conversation: any) => {
        result = [{ id: (conversation.conversation_id ?? conversation.memory_id), name: conversation.name, created_at: conversation.create_time }, ...result]
    });
    setConversations(result)
    if (result.length > 0 && settings.activeConversation == "") {
        setActiveConversation(result[0].id, settings, setSettings, loadActiveConversation);
    }
  }

  return (
    <MantineProvider
      theme={{
        fontFamily: '"Segoe UI",Roboto,Helvetica Neue,Arial,"Noto Sans",sans-serif,"Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol","Noto Color Emoji"',
        fontFamilyMonospace: 'Monaco, Courier, monospace',
        headings: { fontFamily: 'Greycliff CF, sans-serif' },
        fontSizes: {
          xs: '0.8rem',
          sm: '1rem',
          md: '1.1rem',
          lg: '1.2rem',
          xl: '1.4rem',
        }
      }}>
      <AppShell
        padding="md"
        header={
          <Header height={80}
            sx={(theme) => ({
              display: 'flex',
              alignItems: 'center',
              paddingLeft: theme.spacing.md,
              paddingRight: theme.spacing.md,
            })}>
            <Group  w="100%">
              <MediaQuery largerThan="sm" styles={{ display: 'none' }}>
                  <Burger
                    opened={navBarOpened}
                    onClick={() => setNavBarOpened((o) => !o)}
                    size="sm"
                    maw="0.5rem"
                    mr="xl"
                  />
              </MediaQuery>
              <Image width={mobileScreen ? "18em" : "24em"} src="./SycamoreDemoQueryUI_Logo.png" />
              {!mobileScreen && (<Container pos="absolute" right="0rem">
                <Text fz="xs" c="dimmed">index: {settings.openSearchIndex}</Text>
                <Text fz="xs" c="dimmed">llm model: {settings.modelName}</Text>
                <Text fz="xs" c="dimmed">llm model id: {settings.modelId}</Text>
              </Container>) }
            </Group>
          </Header>
        }
        navbarOffsetBreakpoint="sm"
        navbar={
            <ConversationListNavbar navBarOpened={navBarOpened} settings={settings} setSettings={setSettings} setErrorMessage={setErrorMessage} loadingConversation={loadingConversation} loadActiveConversation={loadActiveConversation} conversations={conversations} setConversations={setConversations} refreshConversations={refreshConversations} setChatHistory={setChatHistory} chatInputRef={chatInputRef} setNavBarOpened={setNavBarOpened}></ConversationListNavbar>
        }
        styles={() => ({
          main: { backgroundColor: "white" },
        })}
      >
          <ChatBox chatHistory={chatHistory} searchResults={searchResults} setChatHistory={setChatHistory}
            setSearchResults={setSearchResults} streaming={streaming} setStreaming={setStreaming} setDocsLoading={setDocsLoading}
            setErrorMessage={setErrorMessage} settings={settings} setSettings={setSettings} refreshConversations={refreshConversations} 
            chatInputRef={chatInputRef}/>
      </AppShell >
    </MantineProvider >
  );
}
