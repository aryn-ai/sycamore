import React, { useState } from 'react';
import { ChatBox } from './Chatbox'
import { ControlPanel } from './Controlpanel'
import { ConversationListNavbar } from './ConversationList'
import { DocList } from './Doclist'
import { AppShell, Burger, Container, Footer, Grid, Group, Header, Image, Notification, Stack, Text } from '@mantine/core';
import { SearchResultDocument, Settings, SystemChat, UserChat } from './Types';
import { IconX } from '@tabler/icons-react';
import { getInteractions } from './OpenSearch';




export default function App() {

  const [settings, setSettings] = useState(new Settings());
  const [searchResults, setSearchResults] = useState(new Array<SearchResultDocument>());
  const [streaming, setStreaming] = useState(false);
  const [docsLoading, setDocsLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState(new Array<SystemChat | UserChat>())
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
      if (true) {
        var previousInteractions = new Array<UserChat | SystemChat>()
        // const interactionsData = await interactionsResponse.json();
        const interactionsData = await interactionsResponse;
        console.info("interactionsData: ", interactionsData)
        interactionsData.interactions.forEach((interaction: any) => {
          const systemChat = new SystemChat(
            {
              id: interaction.interaction_id + "_response",
              response: interaction.response,
              interaction_id: interaction.interaction_id,
              // ragPassageCount: null,
              modelName: null,
              queryUsed: null
            });
          const userChat = new UserChat(
            {
              id: interaction.interaction_id + "_user",
              query: interaction.input,
              interaction_id: interaction.interaction_id,
              rephrasedQuery: null
            });
          previousInteractions = [...previousInteractions, systemChat, userChat]
        });
        console.log("Setting previous interactions", previousInteractions)
        setChatHistory(previousInteractions)
      } else {
        console.log("Couldn't load previous interactions for conversation", interactionsResponse)
        setErrorMessage("Couldn't load previous interactions for conversation")
      }
      setLoadingConversation(false)
      setStreaming(false)
    }
    populateConversationMessages();
  }

  return (
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
              // color={theme.colors.gray[6]}
              maw="1rem"
              mr="xl"
            />
            <Image width="14em" src="./SycamoreDemoQueryUI_Logo.png" />


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
      <Grid mah="100vh">

        <Grid.Col span={4} maw="40rem">
          <ChatBox chatHistory={chatHistory} searchResults={searchResults} setChatHistory={setChatHistory}
            setSearchResults={setSearchResults} streaming={streaming} setStreaming={setStreaming} setDocsLoading={setDocsLoading} setErrorMessage={setErrorMessage} settings={settings} />
        </Grid.Col>
        <Grid.Col span={6}>
          <Stack>
            {errorMessage && <ErrorNotification message={errorMessage} />}
            <DocList documents={searchResults} settings={settings} docsLoading={docsLoading} />
          </Stack>
        </Grid.Col>
      </Grid>

    </AppShell >
  );
}