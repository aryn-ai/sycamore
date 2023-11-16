import React, { FormEventHandler, useEffect } from 'react';
import { Dispatch, SetStateAction, useCallback, useRef, useState } from 'react';
import { ActionIcon, Anchor, Button, Card, Center, Code, Container, Divider, Flex, Group, HoverCard, Loader, ScrollArea, Skeleton, Stack, Text, TextInput, useMantineTheme } from '@mantine/core';
import { IconSearch, IconChevronRight, IconRefresh, IconLink, IconFileTypeHtml, IconFileTypePdf } from '@tabler/icons-react';
import { getAnswer, rephraseQuestion } from './Llm';
import { SearchResultDocument, Settings, SystemChat, UserChat } from './Types';
import { hybridConversationSearch, hybridConversationSearchNoRag, queryOpenSearch } from './OpenSearch';

const Citation = ({ document, citationNumber }: { document: SearchResultDocument, citationNumber: number }) => {
    const [doc, setDoc] = useState(document)
    const [docId, setDocId] = useState(document.id)
    const [docUrl, setDocUrl] = useState(document.url)
    const [citNum, setCitNum] = useState(citationNumber)
    const theme = useMantineTheme();
    function icon() {
        if (document.isPdf()) {
            return (<IconFileTypePdf size="1.125rem" color={theme.colors.blue[6]} />)
        } else if (document.url.endsWith("htm") || document.url.endsWith("html")) {
            return (<IconFileTypeHtml size="1.125rem" color={theme.colors.blue[6]} />)
        }
        return (<IconLink size="1.125rem" />)
    };
    return (
        <HoverCard shadow="sm">
            <HoverCard.Target>
                <Anchor key={docId + Math.random()} fz="xs" target="_blank" style={{ "verticalAlign": "super" }} onClick={(event) => {
                    event.preventDefault();
                    if (doc.isPdf()) {
                        const dataString = JSON.stringify(doc);
                        localStorage.setItem('pdfDocumentMetadata', dataString);
                        window.open('/viewPdf');
                    } else {
                        window.open(docUrl);
                    }
                }} >
                    [{citNum}]
                </Anchor>
            </HoverCard.Target>
            <HoverCard.Dropdown>
                <Group>
                    <Text size="xs">
                        {doc.title}
                    </Text>
                    {icon()}
                </Group>
                <Text size="xs" c={theme.colors.gray[6]}> {doc.url}</Text>
            </HoverCard.Dropdown>
        </HoverCard>
    );
}
const LoadingChatBox = ({ loadingMessage }: { loadingMessage: (string | null) }) => {
    const theme = useMantineTheme();
    return (
        <Container ml={theme.spacing.xl} p="lg" miw="80%">
            {/* <Skeleton height={50} circle mb="xl" /> */}
            <Text size="xs" fs="italic" fw="400" p="xs">{loadingMessage ? loadingMessage : null}</Text>
            <Skeleton height={8} radius="xl" />
            <Skeleton height={8} mt={6} radius="xl" />
            <Skeleton height={8} mt={6} width="70%" radius="xl" />
        </Container >
    );
}
const SystemChatBox = ({ systemChat }: { systemChat: SystemChat }) => {
    const citationRegex = /\[(\d+)\]/g;
    const theme = useMantineTheme();
    // const [searchResultsCopy, setSearchResultsCopy] = useState(systemChat.hits)
    const replaceCitationsWithLinks = (text: string) => {
        const elements: React.ReactNode[] = new Array();
        var lastIndex = 0;
        if (text == null)
            return elements;
        text.replace(citationRegex, (substring: string, citationNumberRaw: any, index: number) => {
            elements.push(text.slice(lastIndex, index));
            const citationNumber = parseInt(citationNumberRaw)
            if (citationNumber >= 1 && citationNumber <= systemChat.hits.length) {
                elements.push(
                    <Citation key={citationNumber} document={systemChat.hits[citationNumber - 1]} citationNumber={citationNumber} />
                );
            } else {
                // elements.push(substring)
            };
            lastIndex = index + substring.length;
            return substring;
        });
        elements.push(text.slice(lastIndex));
        return elements;
    };
    // setTextNodes(replaceCitationsWithLinks(systemChat.response))
    return (
        <Card key={systemChat.id} ml={theme.spacing.xl} padding="lg" radius="md" bg="blue.0">
            <Text size="sm" sx={{ "whiteSpace": "pre-wrap" }}>
                {/* {textNodes} */}
                {replaceCitationsWithLinks(systemChat.response)}
            </Text>
            {systemChat.ragPassageCount ?
                <Container>
                    <Divider size={"xs"} mt="md" mb="sm" />

                    <Text size="sm" fs="italic" fw="500" align='right'>
                        {systemChat.queryUsed}
                    </Text>
                    <Text size="xs" color="dimmed" align='right'>
                        RAG passage count: {systemChat.ragPassageCount} Model used: {systemChat.modelName}
                    </Text></Container>
                : null}
            <Text fz="xs" fs="italic" color="dimmed" align='right'>
                Interaction id: {systemChat.interaction_id ? systemChat.interaction_id : "[todo]"}
            </Text>
        </Card >
    );
}

const UserChatBox = ({ id, interaction_id, query, rephrasedQuery }: UserChat) => {
    const theme = useMantineTheme();

    return (
        <Card key={id} mr={theme.spacing.xl} padding="lg">
            <Text size="sm" fw={500}>
                {query}
            </Text>{rephrasedQuery ?
                <Text fz="xs" fs="italic" color="dimmed">
                    Rephrased question: {rephrasedQuery}
                </Text>
                : null}
            <Text fz="xs" fs="italic" color="dimmed">
                Interaction id: {interaction_id ? interaction_id : "[todo]"}
            </Text>
        </Card>
    );
}

function parseOpenSearchResults(openSearchResponse: any, setErrorMessage: Dispatch<SetStateAction<string | null>>) {
    if ((openSearchResponse.error !== undefined) &&
        (openSearchResponse.error.type === 'timeout_exception')) {
        const documents = new Array<SearchResultDocument>()
        const chatResponse = "Timeout from OpenAI"
        const interactionId = ""
        setErrorMessage(chatResponse)
        return {
            documents: documents,
            chatResponse: chatResponse,
            interactionId: interactionId
        }
    }
    const documents = openSearchResponse.hits.hits.map((result: any, idx: number) => {
        const doc = result._source
        return new SearchResultDocument({
            id: result._id,
            index: idx + 1,
            title: doc.properties.title ?? "Untitled",
            description: doc.text_representation,
            url: doc.properties._location ?? doc.properties.path,
            relevanceScore: "" + result._score,
            properties: doc.properties,
            bbox: doc.bbox
        });
    });
    const chatResponse = openSearchResponse.ext.retrieval_augmented_generation.answer
    const interactionId = openSearchResponse.ext.retrieval_augmented_generation.interaction_id
    return {
        documents: documents,
        chatResponse: chatResponse,
        interactionId: interactionId
    }
}

function parseOpenSearchResultsOg(openSearchResponse: any) {
    const documents = openSearchResponse.hits.hits.map((result: any, idx: number) => {
        const doc = result._source
        return new SearchResultDocument({
            id: result._id,
            index: idx + 1,
            title: doc.properties.title ?? "Untitled",
            description: doc.text_representation,
            url: doc.properties._location ?? doc.properties.path,
            relevanceScore: "" + result._score,
            properties: doc.properties,
            bbox: doc.bbox
        });
    });
    return documents
}

export const ChatBox = ({ chatHistory, searchResults, setChatHistory, setSearchResults, streaming, setStreaming, setDocsLoading, setErrorMessage, settings }:
    {
        chatHistory: (UserChat | SystemChat)[], searchResults: SearchResultDocument[], setChatHistory: Dispatch<SetStateAction<any[]>>,
        setSearchResults: Dispatch<SetStateAction<any[]>>, streaming: boolean, setStreaming: Dispatch<SetStateAction<boolean>>,
        setDocsLoading: Dispatch<SetStateAction<boolean>>, setErrorMessage: Dispatch<SetStateAction<string | null>>, settings: Settings
    }) => {
    const theme = useMantineTheme();
    const chatInputRef = useRef<HTMLInputElement | null>(null);
    const [chatInput, setChatInput] = useState("");
    const [loadingMessage, setLoadingMessage] = useState<string | null>(null);

    // This method does all the search workflow execution
    const handleSubmitParallelDocLoad = async (e: React.FormEvent) => {
        try {
            e.preventDefault();
            if (chatInputRef.current != null) {
                chatInputRef.current.disabled = true
            }

            setStreaming(true);
            setDocsLoading(true)
            console.log("Rephrasing question: ", chatInput)
            // Generate conversation text list
            const chatHistoryInteractions = chatHistory.map((chat) => {
                if ('query' in chat) {
                    return { role: "user", content: chat.query }
                } else {
                    return { role: "system", content: chat.response ?? "" }
                }
            })
            // console.log("history: ", chatHistoryText)
            setLoadingMessage("Rephrasing question with conversation context");
            const rephraseQuestionResponse = await rephraseQuestion(chatInput, chatHistoryInteractions, settings.modelName)
            if (rephraseQuestionResponse.ok) {
                const responseData = await rephraseQuestionResponse.json();
                const rephrasedQuestion = responseData.choices[0].message.content;
                console.log("Rephrased question to ", rephrasedQuestion)
                setLoadingMessage("Querying knowledge database with rephrased question: \"" + rephrasedQuestion + "\"");

                const populateChatFromOs = (openSearchResults: any) => {
                    console.log("Main processor ", openSearchResults)
                    console.log("Main processor: OS results ", openSearchResults)
                    const endTime = new Date(Date.now());
                    const elpased = endTime.getTime() - startTime.getTime()
                    console.log("Main processor: OS took seconds: ", elpased)
                    const parsedOpenSearchResults = parseOpenSearchResults(openSearchResults, setErrorMessage)
                    // todo: replace these with interaction id
                    const newChat = new UserChat({
                        id: parsedOpenSearchResults.interactionId + "_user",
                        interaction_id: parsedOpenSearchResults.interactionId,
                        query: chatInput,
                        rephrasedQuery: rephrasedQuestion
                    });
                    const newSystemChat = new SystemChat(
                        {
                            id: parsedOpenSearchResults.interactionId + "_system",
                            interaction_id: parsedOpenSearchResults.interactionId,
                            response: parsedOpenSearchResults.chatResponse,
                            ragPassageCount: settings.ragPassageCount,
                            modelName: settings.modelName,
                            queryUsed: rephrasedQuestion,
                            hits: parsedOpenSearchResults.documents
                        });
                    setChatHistory([newSystemChat, newChat, ...chatHistory,]);
                }
                const populateDocsFromOs = (openSearchResults: any) => {
                    console.log("Info separate processor ", openSearchResults)
                    console.log("Info separate processor : OS results ", openSearchResults)
                    const endTime = new Date(Date.now());
                    const elpased = endTime.getTime() - startTime.getTime()
                    console.log("Info separate processor : OS took seconds: ", elpased)
                    const parsedOpenSearchResults = parseOpenSearchResultsOg(openSearchResults)
                    setSearchResults(parsedOpenSearchResults)
                    console.log("Info separate processor : set docs in independent thread to: ", parsedOpenSearchResults)
                    setDocsLoading(false)
                }


                const startTime = new Date(Date.now());
                // const openSearchResults = await hybridConversationSearch(chatInput, rephrasedQuestion, settings.activeConversation, settings.openSearchIndex, settings.modelName, settings.ragPassageCount);
                await Promise.all([
                    hybridConversationSearchNoRag(rephrasedQuestion, settings.openSearchIndex, settings.embeddingModel).then(populateDocsFromOs),
                    hybridConversationSearch(chatInput, rephrasedQuestion, settings.activeConversation, settings.openSearchIndex, settings.embeddingModel, settings.modelName, settings.ragPassageCount).then(populateChatFromOs),
                ]);

            } else {
                console.error('Error calling the API:', rephraseQuestionResponse.statusText);
                throw new Error('Error calling the API: ' + rephraseQuestionResponse.statusText);
            }
        } catch (e) {
            if (typeof e === "string") {
                setErrorMessage(e.toUpperCase())
            } else if (e instanceof Error) {
                setErrorMessage(e.message)
            }
        } finally {
            setStreaming(false);
            setChatInput("");
            setDocsLoading(false);
            setLoadingMessage(null);
            if (chatInputRef.current != null) {
                // chatInputRef.current.focus();
                chatInputRef.current.disabled = false
            }
        }
    }

    // This method does all the search workflow execution
    const handleSubmit = async (e: React.FormEvent) => {
        return handleSubmitParallelDocLoad(e)
        // try {

        //     e.preventDefault();
        //     if (chatInputRef.current != null) {
        //         chatInputRef.current.disabled = true
        //     }

        //     setStreaming(true);
        //     setDocsLoading(true)
        //     console.log("Rephrasing question: ", chatInput)
        //     // Generate conversation text list
        //     const chatHistoryInteractions = chatHistory.map((chat) => {
        //         if ('query' in chat) {
        //             return { role: "user", content: chat.query }
        //         } else {
        //             return { role: "system", content: chat.response }
        //         }
        //     })
        //     // console.log("history: ", chatHistoryText)
        //     const rephraseQuestionResponse = await rephraseQuestion(chatInput, chatHistoryInteractions, settings.modelName)
        //     if (rephraseQuestionResponse.ok) {
        //         const responseData = await rephraseQuestionResponse.json();
        //         const rephrasedQuestion = responseData.choices[0].message.content;
        //         console.log("Rephrased question to ", rephrasedQuestion)
        //         setLoadingMessage("Query to knowledge database: " + rephrasedQuestion);

        //         // console.log("Starting og path")
        //         // const startTimeOg = new Date(Date.now());
        //         // const openSearchResultsOg = await queryOpenSearch(rephrasedQuestion, settings.openSearchIndex);
        //         // const parsedOpenSearchResultsOg = parseOpenSearchResultsOg(openSearchResultsOg)
        //         // console.log("Starting og LLM path")
        //         // const llmResponse = await getAnswer(chatInput, parsedOpenSearchResultsOg.slice(0, settings.ragPassageCount))
        //         // const endTimeOg = new Date(Date.now());
        //         // const elpasedOg = endTimeOg.getTime() - startTimeOg.getTime()
        //         // console.log("OG took seconds: ", elpasedOg)


        //         const startTime = new Date(Date.now());
        //         const openSearchResults = await hybridConversationSearch(chatInput, rephrasedQuestion, settings.activeConversation, settings.openSearchIndex, settings.modelName, settings.ragPassageCount);

        //         console.log("OS results ", openSearchResults)
        //         const endTime = new Date(Date.now());
        //         const elpased = endTime.getTime() - startTime.getTime()
        //         console.log("OS took seconds: ", elpased)
        //         const parsedOpenSearchResults = parseOpenSearchResults(openSearchResults)


        //         setSearchResults(parsedOpenSearchResults.documents)
        //         console.log("set docs to: ", parsedOpenSearchResults.documents)
        //         setDocsLoading(false)

        //         // todo: replace these with interaction id
        //         const newChat = new UserChat({
        //             id: parsedOpenSearchResults.interactionId + "_user",
        //             interaction_id: parsedOpenSearchResults.interactionId,
        //             query: chatInput,
        //             rephrasedQuery: rephrasedQuestion
        //         });
        //         const newSystemChat = new SystemChat(
        //             {
        //                 id: parsedOpenSearchResults.interactionId + "_system",
        //                 interaction_id: parsedOpenSearchResults.interactionId,
        //                 response: parsedOpenSearchResults.chatResponse,
        //                 ragPassageCount: settings.ragPassageCount,
        //                 modelName: settings.modelName,
        //                 queryUsed: rephrasedQuestion,
        //                 hits: parsedOpenSearchResults.documents
        //             });
        //         setChatHistory([newSystemChat, newChat, ...chatHistory,]);

        //         // if (llmResponse.ok) {
        //         //     const llmAnswerData = await llmResponse.json();
        //         //     const llmAnswer = llmAnswerData.choices[0].message.content;
        //         //     console.info("LLM answer is ", llmAnswer)
        //         //     console.info("llm data is ", llmAnswerData)
        //         //     const newSystemChat = new SystemChat(
        //         //         {
        //         //             id: "some another id",
        //         //             response: llmAnswer,
        //         //             ragPassageCount: settings.ragPassageCount,
        //         //             modelName: settings.modelName,
        //         //             queryUsed: rephrasedQuestion
        //         //         });
        //         //     setChatHistory([newSystemChat, newChat, ...chatHistory,]);
        //         // } else {
        //         //     console.error('Error calling the API:', llmResponse.statusText);
        //         //     throw new Error('Error calling the API: ' + llmResponse.statusText);
        //         // }
        //     } else {
        //         console.error('Error calling the API:', rephraseQuestionResponse.statusText);
        //         throw new Error('Error calling the API: ' + rephraseQuestionResponse.statusText);
        //     }
        // } catch (e) {
        //     if (typeof e === "string") {
        //         setErrorMessage(e.toUpperCase())
        //     } else if (e instanceof Error) {
        //         setErrorMessage(e.message)
        //     }
        // } finally {
        //     setStreaming(false);
        //     setChatInput("");
        //     setDocsLoading(false);
        //     setLoadingMessage(null);
        //     if (chatInputRef.current != null) {
        //         // chatInputRef.current.focus();
        //         chatInputRef.current.disabled = false
        //     }
        // }
    };

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setChatInput(e.target.value);
    };

    const handleInputKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            handleSubmit(e);
        }
    };
    React.useEffect(() => {
        chatInputRef.current?.focus();
    }, [streaming]);
    return (
        <Flex direction="column" h="90vh" sx={{ 'borderStyle': 'none solid none none', 'borderColor': '#eee;' }}>
            <form onSubmit={handleSubmit} className="input-form">
                <TextInput
                    onKeyDown={handleInputKeyPress}
                    onChange={handleInputChange}
                    ref={chatInputRef}
                    value={chatInput}
                    icon={<IconSearch size="1.1rem" stroke={1.5} />}
                    radius="xl"
                    autoFocus
                    size="sm"
                    fz="xs"
                    p="sm"
                    rightSection={
                        <ActionIcon size={32} radius="xl" bg="#5688b0" variant="filled">
                            <IconChevronRight size="1rem" stroke={2} onClick={handleSubmit} />
                        </ActionIcon>
                    }
                    placeholder="Ask me anything"
                    rightSectionWidth={42}
                    disabled={settings.activeConversation == null}
                />
            </form>
            {/* <Center></Center> */}
            <Center>
                <Text fz="xs" color="dimmed">
                    Active conversation: {settings.activeConversation ? settings.activeConversation : "None"}

                </Text>
            </Center>
            {loadingMessage ? <LoadingChatBox loadingMessage={loadingMessage} /> : null}
            {/* <LoadingChatBox loadingMessage={loadingMessage} /> */}
            <Center>
                {streaming ? <Loader size="xs" variant="dots" m="md" /> : ""}
            </Center>
            <ScrollArea.Autosize bg="white" p="md" mah="calc(90vh - 12rem);" mb="md">
                <Stack>

                    {chatHistory.map((chat, index) => {
                        if ('query' in chat) {
                            return <UserChatBox key={chat.id + "_user"} id={chat.id} interaction_id={chat.interaction_id} query={chat.query} rephrasedQuery={chat.rephrasedQuery} />
                        } else {
                            return <SystemChatBox key={chat.id + "_system"} systemChat={chat} />
                        }
                    }
                    )
                    }
                </Stack>
            </ScrollArea.Autosize>

        </Flex >
    );
}
