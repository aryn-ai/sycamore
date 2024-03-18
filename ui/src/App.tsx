import React, { useState } from 'react';
import { ChatBox, thumbToBool } from './Chatbox'
import { ControlPanel } from './Controlpanel'
import { ConversationListNavbar } from './ConversationList'
import { DocList } from './Doclist'
import { AppShell, Burger, Container, Footer, Grid, Group, Header, Image, MantineProvider, Notification, Stack, Text } from '@mantine/core';
import { SearchResultDocument, Settings, SystemChat, UserChat } from './Types';
import { IconX } from '@tabler/icons-react';
import { getFeedback, getInteractions } from './OpenSearch';




export default function App() {

  const [settings, setSettings] = useState(new Settings());
  const [searchResults, setSearchResults] = useState(new Array<SearchResultDocument>());
  const [streaming, setStreaming] = useState(false);
  const [docsLoading, setDocsLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState(new Array<SystemChat>())
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [navBarOpened, setNavBarOpened] = useState(true);
  const [loadingConversation, setLoadingConversation] = useState(false);

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
        previousInteractions = [...previousInteractions, chat]
      })
      console.log("Setting previous interactions", previousInteractions)
      setChatHistory(previousInteractions)
      setLoadingConversation(false)
      setStreaming(false)
    }
    populateConversationMessages();
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
            <Group grow w="100%">

              <Burger
                opened={navBarOpened}
                onClick={() => setNavBarOpened((o) => !o)}
                size="xs"
                maw="1rem"
                mr="xl"
              />
              <Image width="24em" src="./SycamoreDemoQueryUI_Logo.png" />


              <Container pos="absolute" right="0rem">
                <Text fz="xs" c="dimmed">index: {settings.openSearchIndex}</Text>
                <Text fz="xs" c="dimmed">llm model: {settings.modelName}</Text>
                <Text fz="xs" c="dimmed">llm model id: {settings.modelId}</Text>
              </Container>
            </Group>
          </Header>
        }
        navbar={
          <ConversationListNavbar navBarOpened={navBarOpened} settings={settings} setSettings={setSettings} setErrorMessage={setErrorMessage} loadingConversation={loadingConversation} loadActiveConversation={loadActiveConversation}></ConversationListNavbar>
        }
        footer={
          < Footer height={60} p="md" >
            <ControlPanel settings={settings} setSettings={setSettings} reset={reset} />
          </Footer >
        }
        styles={() => ({
          main: { backgroundColor: "white" },
        })}
      >
        <Container>

          <ChatBox chatHistory={chatHistory} searchResults={searchResults} setChatHistory={setChatHistory}
            setSearchResults={setSearchResults} streaming={streaming} setStreaming={setStreaming} setDocsLoading={setDocsLoading}
            setErrorMessage={setErrorMessage} settings={settings} setSettings={setSettings} />
        </Container>

      </AppShell >
    </MantineProvider >
  );
}
