import React, { useEffect } from 'react';
import { Dispatch, SetStateAction, useRef, useState } from 'react';
import { ActionIcon, Anchor, Badge, Box, Button, Card, Center, Chip, Container, Flex, Group, HoverCard, Image, JsonInput, Loader, Modal, NativeSelect, ScrollArea, Skeleton, Stack, Switch, Text, TextInput, Title, Tooltip, UnstyledButton, createStyles, useMantineTheme } from '@mantine/core';
import { IconSearch, IconChevronRight, IconLink, IconFileTypeHtml, IconFileTypePdf, IconX, IconEdit, IconPlayerPlayFilled, IconPlus, IconSettings, IconInfoCircle, IconWriting } from '@tabler/icons-react';
import { IconThumbUp, IconThumbUpFilled, IconThumbDown, IconThumbDownFilled } from '@tabler/icons-react';
import { getFilters, rephraseQuestion } from './Llm';
import { SearchResultDocument, Settings, SystemChat } from './Types';
import { hybridConversationSearch, updateInteractionAnswer, updateFeedback, getHybridConversationSearchQuery, openSearchCall, createConversation, hybridSearchNoRag } from './OpenSearch';
import { DocList } from './Doclist';
import { useDisclosure, useMediaQuery } from '@mantine/hooks';
import { Prism } from '@mantine/prism';
import { ControlPanel } from './Controlpanel';

const useStyles = createStyles((theme) => ({

    inputBar: {
        width: '50vw',
        [theme.fn.smallerThan('sm')]: {
            width: '100%',
        },
    },
    fixedBottomContainer: {
        margin: 0,
        maxWidth: 'none',
        bottom: 0,
        borderTop: '1px solid lightgrey',
        flex: 1,
    },
    settingsIcon: {
        // left,
        
        zIndex: 1
    },
    chatHistoryContainer: {
        height: `calc(100vh - 14.5em)`,
    },
    settingsStack: {
        position: 'absolute',
        top: 10,
        right: 0,
        alignItems: 'end'
    },
    chatFlex: {
        paddingBottom: 0,
    }
}))


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
                    <Badge size="xs" color="gray" variant="filled">{citNum}</Badge>
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
const FilterInput = ({ settings, filtersInput, setFiltersInput, filterError, setFilterError }: { settings: Settings, filtersInput: any, setFiltersInput: any, filterError: boolean, setFilterError: any }) => {
    const handleInputChange = (filterName: string, value: string) => {
        if(filterError) {
            setFilterError(false);
        }
        setFiltersInput((prevValues: any) => ({
            ...prevValues,
            [filterName]: value,
        }));
    };

    return (
        <Group spacing="0">
            {
                settings.required_filters.map(required_filter => (
                    <Group spacing="0">
                        <Text size="xs">{required_filter}</Text>
                        <TextInput
                            onChange={(e) => handleInputChange(required_filter, e.target.value)}
                            value={filtersInput[required_filter] || ''}
                            autoFocus
                            required
                            error={filterError }
                            size="xs"
                            fz="xs"
                            pl="sm"
                        />
                    </Group>
                ))
            }
        </Group>
    )
}
const SearchControlPanel = ({ disableFilters, setDisableFilters, questionRewriting, setQuestionRewriting, queryPlanner, setQueryPlanner, chatHistory, setChatHistory, openSearchQueryEditorOpenedHandlers, settings }:
    { disableFilters: any, setDisableFilters: any, questionRewriting: any, setQuestionRewriting: any, queryPlanner: boolean, setQueryPlanner: any, chatHistory: any, setChatHistory: any, openSearchQueryEditorOpenedHandlers: any, settings: Settings }) => {
    return (
        <Group position='right' w="100%">
            {((settings.required_filters.length > 0) ?
                <Chip size="xs" checked={!disableFilters} onChange={() => setDisableFilters((v: any) => !v)} variant="light">
                    Filters
                </Chip> : null)
            }
            {
                settings.auto_filter ?
                    <Chip key="queryPlanner" size="xs" checked={queryPlanner} onChange={() => setQueryPlanner(!queryPlanner)} variant="light">
                        Auto-filters
                    </Chip> : null
            }
            <Button compact onClick={() => openSearchQueryEditorOpenedHandlers.open()} size="xs" fz="xs" variant="gradient" gradient={{ from: 'blue', to: 'indigo', deg: 90 }}>
                Run OpenSearch Query
            </Button>
        </Group >
    )
}

const FeedbackButtons = ({ systemChat, settings }: { systemChat: SystemChat, settings: Settings }) => {
    const [thumbUpState, setThumbUp] = useState(systemChat.feedback);
    const [comment, setComment] = useState(systemChat.comment);
    const handleSubmit = async (thumb: boolean | null) => {
        updateFeedback(settings.activeConversation, systemChat.interaction_id, thumb, comment)
    };
    const handleInputKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter') {
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
                <ActionIcon size={32} radius="xs" component="button"
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
                    }}>
                    {thumbUpState == null || !thumbUpState ?
                        <IconThumbUp size="1rem" /> :
                        <IconThumbUpFilled size="1rem" color="green" fill="green" />
                    }
                </ActionIcon>
                <ActionIcon size={32} radius="xs" component="button"
                    onClick={(event) => {
                        if (thumbUpState == null || thumbUpState) {
                            setThumbUp(false);
                            systemChat.feedback = false;
                            handleSubmit(false);
                            // } else if (systemChat.feedback == false) {
                        } else {
                            setThumbUp(null);
                            systemChat.feedback = null;
                            handleSubmit(null);
                        }
                    }}>
                    {thumbUpState == null || thumbUpState ?
                        <IconThumbDown size="1rem" /> :
                        <IconThumbDownFilled size="1rem" color="red" fill="red" />
                    }
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


const OpenSearchQueryEditor = ({ openSearchQueryEditorOpened, openSearchQueryEditorOpenedHandlers, currentOsQuery, currentOsUrl, setCurrentOsQuery, setCurrentOsUrl, setLoadingMessage, chatHistory, setChatHistory }:
    { openSearchQueryEditorOpened: boolean, openSearchQueryEditorOpenedHandlers: any, currentOsQuery: string, currentOsUrl: string, setCurrentOsQuery: any, setCurrentOsUrl: any, setLoadingMessage: any, chatHistory: any, setChatHistory: any }) => {

    const handleOsSubmit = (e: React.MouseEvent<HTMLButtonElement>) => {
        runJsonQuery(currentOsQuery, currentOsUrl);
    };


    // json query run
    const runJsonQuery = async (newOsJsonQuery: string, currentOsQueryUrl: string) => {
        try {

            openSearchQueryEditorOpenedHandlers.close()
            setLoadingMessage("Processing query...")

            const query = JSON.parse(newOsJsonQuery)
            const populateChatFromRawOsQuery = async (openSearchResults: any) => {
                const openSearchResponse = await openSearchResults
                // send question and OS results to LLM
                const response = await interpretOsResult(newOsJsonQuery, JSON.stringify(openSearchResponse, null, 4))
                var length = 10
                const newSystemChat = new SystemChat(
                    {
                        id: Math.random().toString(36).substring(2, length + 2),
                        response: response,
                        queryUsed: "User provided OpenSearch query",
                        rawQueryUsed: newOsJsonQuery,
                        queryUrl: currentOsQueryUrl,
                        rawResults: openSearchResponse,
                        interaction_id: "Adhoc, not stored in memory",
                        editing: false,
                        hits: []
                    });
                setChatHistory([...chatHistory, newSystemChat]);
            }
            const startTime = new Date(Date.now());
            await Promise.all([
                openSearchCall(query, currentOsQueryUrl)
                    .then(populateChatFromRawOsQuery),
            ]);
        } finally {
            setLoadingMessage(null)
        }
    }

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setCurrentOsUrl(e.target.value);
    };

    const jsonPlaceholder = {
        "query": {
            "match_all": {}
        },
        "size": 300
    }
    return (
        <Modal opened={openSearchQueryEditorOpened} onClose={openSearchQueryEditorOpenedHandlers.close} title="OpenSearch Query Editor" size="calc(80vw - 3rem)">
                <Text fz="xs" p="sm">Note: If you want a RAG answer, make sure the search pipeline is being used.
                    Ensure it's configured in the URL (search_pipeline=hybrid_rag_pipeline), and also in the query itself (ext.generative_qa_parameters parameters)</Text>
                <Group position="apart" grow>
                    <TextInput
                        size="xs"
                        value={currentOsUrl}
                        onChange={handleInputChange}
                        label="OpenSearch url"
                        placeholder='e.g. /opensearch/myindex/_search?'
                        p="sm"
                        withAsterisk
                    />
                    <Button maw="5rem" fz="xs" size="xs" m="md" color="teal" leftIcon={<IconPlayerPlayFilled size="0.6rem" />} onClick={e => handleOsSubmit(e)} >
                        Run
                    </Button>
                </Group>
                <ScrollArea >
                    <JsonInput
                        value={currentOsQuery}
                        onChange={newValue => setCurrentOsQuery(newValue)}
                        validationError="Invalid JSON"
                        placeholder={"e.g.\n" + JSON.stringify(jsonPlaceholder, null, 4)}
                        formatOnBlur
                        autosize
                        minRows={4}
                    />
                </ScrollArea>
        </Modal>
    )
};

/**
 * This component manages an interaction effectively. It shows the question/answer/hits, and also supports the edit/resubmit functionality.
 * All context here is lost when switching a conversation or refreshing the page.
 */
const SystemChatBox = ({ systemChat, chatHistory, settings, handleSubmit, setChatHistory, setSearchResults, setErrorMessage, setLoadingMessage, setCurrentOsQuery, setCurrentOsUrl, openSearchQueryEditorOpenedHandlers, disableFilters }:
    { systemChat: SystemChat, chatHistory: any, settings: Settings, handleSubmit: any, setChatHistory: any, setSearchResults: any, setErrorMessage: any, setLoadingMessage: any, setCurrentOsQuery: any, setCurrentOsUrl: any, openSearchQueryEditorOpenedHandlers: any, disableFilters: boolean }) => {
    const citationRegex = /\[(\d+)\]/g;
    const theme = useMantineTheme();
    console.log("Filter content is", systemChat.filterContent)
    const replaceCitationsWithLinks = (text: string) => {
        let cleanedText = text.replace(/\[\${(\d+)}\]/g, "[$1]").replace(/\\n/g, "\n"); //handle escaped strings
        const elements: React.ReactNode[] = new Array();
        var lastIndex = 0;
        if (text == null)
            return elements;
        cleanedText.replace(citationRegex, (substring: string, citationNumberRaw: any, index: number) => {
            elements.push(cleanedText.slice(lastIndex, index));
            const citationNumber = parseInt(citationNumberRaw)
            if (citationNumber >= 1 && citationNumber <= systemChat.hits.length) {
                elements.push(
                    <Citation key={citationNumber} document={systemChat.hits[citationNumber - 1]} citationNumber={citationNumber} />
                );
            } else {
                elements.push(substring)
            };
            lastIndex = index + substring.length;
            return substring;
        });
        elements.push(cleanedText.slice(lastIndex));
        return elements;
    };

    // for editing
    const { query, url } = getHybridConversationSearchQuery(systemChat.queryUsed, systemChat.queryUsed, parseFilters(systemChat.filterContent ?? {}, setErrorMessage), settings.openSearchIndex, settings.embeddingModel, settings.modelName, settings.ragPassageCount)
    const queryUrl = systemChat.queryUrl != "" ? systemChat.queryUrl : url
    const currentOsQuery = systemChat.rawQueryUsed != null && systemChat.rawQueryUsed != "" ? systemChat.rawQueryUsed : JSON.stringify(
        query,
        null,
        4
    )
    const [editing, setEditing] = useState(systemChat.editing)
    const [newQuestion, setNewQuestion] = useState(systemChat.queryUsed)
    const [newFilterContent, setNewFilterContent] = useState(systemChat.filterContent ?? {})
    const [newFilterInputDialog, newFilterInputDialoghHandlers] = useDisclosure(false);

    const [newFilterType, setNewFilterType] = useState("location")
    const [newFilterValue, setNewFilterValue] = useState("")


    const filters = () => {
        if (systemChat.filterContent == null && !editing) {
            return null;
        }

        const removeFilter = (filterToRemove: any) => {
            console.log("Removing filter: ", filterToRemove)
            const updatedNewFilterContent = { ...newFilterContent }
            delete updatedNewFilterContent[filterToRemove];
            setNewFilterContent(updatedNewFilterContent);
        };

        const editFilter = (filterToEdit: any) => {
            setNewFilterType(filterToEdit)
            setNewFilterValue(newFilterContent[filterToEdit])
            newFilterInputDialoghHandlers.open()
        };

        const addFilter = () => {

            const handleSubmit = (e: React.FormEvent) => {
                console.log("Adding filter", newFilterType + " " + newFilterValue)
                const updatedNewFilterContent = { ...newFilterContent }
                updatedNewFilterContent[newFilterType] = newFilterValue
                setNewFilterContent(updatedNewFilterContent);
                newFilterInputDialoghHandlers.close()
            };

            const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
                setNewFilterValue(e.target.value);
            };

            const handleInputKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    handleSubmit(e);
                }
            };

            return (
                <Modal opened={newFilterInputDialog} onClose={newFilterInputDialoghHandlers.close} title="Add filter" size="auto">
                    <Container p="md">
                        <NativeSelect
                            value={newFilterType}
                            label="Field to filter on"
                            onChange={(event) => setNewFilterType(event.currentTarget.value)}
                            data={[
                                { value: 'location', label: 'Location' },
                                { value: 'airplane_name', label: 'Airplane type' },
                                { value: 'date_start', label: 'Before' },
                                { value: 'date_end', label: 'After' },
                            ]}
                        />
                        <TextInput
                            label="Value of filter"
                            value={newFilterValue}
                            onKeyDown={handleInputKeyPress}
                            onChange={handleInputChange}
                            autoFocus
                            size="sm"
                            fz="xs"
                            placeholder="e.g. California"
                        />
                        <Button onClick={(e) => handleSubmit(e)}>
                            Add filter
                        </Button>
                    </Container>
                </Modal>
            )

        };

        const editFiltersButtons = (filter: any) => (
            <Group position="right" spacing="0">
                <ActionIcon size="0.8rem" color="blue" radius="md" variant="transparent" onClick={() => editFilter(filter)}>
                    <IconEdit size="xs" />
                </ActionIcon>
                <ActionIcon size="0.8rem" color="blue" radius="md" variant="transparent" onClick={() => removeFilter(filter)}>
                    <IconX size="xs" />
                </ActionIcon>
            </Group>
        );

        if (!editing) {
            return (
                <Container mb="sm">
                    {
                        Object.keys(systemChat.filterContent).map((filter: any) => {
                            if (systemChat.filterContent[filter] != "unknown") {
                                return (
                                    <Badge size="xs" key={filter} p="xs" mr="xs" >
                                        {filter}: {systemChat.filterContent[filter]}
                                    </Badge>
                                )
                            }
                        }
                        )
                    }
                </Container >
            )
        } else {
            return (

                <Container mb="sm">
                    {addFilter()}
                    {
                        Object.keys(newFilterContent).map((filter: any) => {
                            if (newFilterContent[filter] != "unknown") {
                                return (
                                    <Badge size="xs" key={filter} p="xs" mr="xs" rightSection={editFiltersButtons(filter)} >
                                        {filter}: {newFilterContent[filter]}
                                    </Badge >
                                )
                            }
                        }
                        )
                    }
                    <UnstyledButton onClick={() => newFilterInputDialoghHandlers.open()}>
                        <Badge size="xs" key="newFilter" p="xs" mr="xs" onClick={() => { setEditing(false); newFilterInputDialoghHandlers.open() }}>
                            <Group>
                                New filter
                                <ActionIcon size="0.8rem" color="blue" radius="md" variant="transparent">
                                    <IconPlus size="xs" />
                                </ActionIcon>
                            </Group>
                        </Badge >
                    </UnstyledButton>
                </Container >
            )
        }
    }
    const handleInputChange = (e: any) => {
        setNewQuestion(e.target.value);
    };

    const handleInputKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            rerunQuery();
        }
    };

    // rerun mechanism
    const rerunQuery = async () => {
        try {
            setEditing(false);
            if(disableFilters) {
                setLoadingMessage("Processing query...")
            }
            else {
                setLoadingMessage("Processing query with filters...")
            }
            const populateChatFromOs = ({ openSearchResponse, query }: { openSearchResponse: any, query: any }) => {
                console.log("New filter content is", newFilterContent)
                console.log("Main processor ", openSearchResponse)
                console.log("Main processor: OS results ", openSearchResponse)
                const endTime = new Date(Date.now());
                const elpased = endTime.getTime() - startTime.getTime()
                console.log("Main processor: OS took seconds: ", elpased)
                const parsedOpenSearchResults = parseOpenSearchResults(openSearchResponse, setErrorMessage)
                const newSystemChat = new SystemChat(
                    {
                        id: parsedOpenSearchResults.interactionId + "_system",
                        interaction_id: parsedOpenSearchResults.interactionId,
                        response: parsedOpenSearchResults.chatResponse,
                        ragPassageCount: settings.ragPassageCount,
                        modelName: settings.modelName,
                        rawQueryUsed: JSON.stringify(query, null, 4),
                        queryUrl: queryUrl,
                        queryUsed: newQuestion,
                        hits: parsedOpenSearchResults.documents,
                        filterContent: newFilterContent
                    });
                setChatHistory([...chatHistory, newSystemChat]);
            }


            const clean = async (result: any) => {
                const openSearchResponseAsync = result[0]
                const query = result[1]
                const openSearchResponse = await openSearchResponseAsync
                let generatedAnswer = openSearchResponse.ext.retrieval_augmented_generation.answer
                if (settings.simplify && openSearchResponse.hits.hits.length > 0) {
                    console.log("Simplifying answer: ", generatedAnswer)
                    generatedAnswer = await simplifyAnswer(newQuestion, generatedAnswer)
                }
                await updateInteractionAnswer(openSearchResponse.ext.retrieval_augmented_generation.interaction_id, generatedAnswer, query)
                openSearchResponse.ext.retrieval_augmented_generation.answer = generatedAnswer
                return { openSearchResponse, query }
            }

            const clean_rag = async ({ openSearchResponse, query }: { openSearchResponse: any, query: any }) => {
                let generatedAnswer = openSearchResponse.ext.retrieval_augmented_generation.answer
                if (settings.simplify && openSearchResponse.hits.hits.length > 0) {
                    console.log("Simplifying answer: ", generatedAnswer)
                    generatedAnswer = await simplifyAnswer(newQuestion, generatedAnswer)
                }
                await updateInteractionAnswer(openSearchResponse.ext.retrieval_augmented_generation.interaction_id, generatedAnswer, query)
                openSearchResponse.ext.retrieval_augmented_generation.answer = generatedAnswer
                return { openSearchResponse, query }
            }

            const anthropic_rag = async (result: any) => {
                const openSearchResponseAsync = result[0]
                const query = result[1]
                const openSearchResponse = await openSearchResponseAsync
                let generatedAnswer = "Error"
                if (openSearchResponse.hits.hits.length > 0) {
                    console.log("Anthropic RAG time...")
                    generatedAnswer = await anthropicRag(newQuestion, openSearchResponse)
                }
                openSearchResponse["ext"] = {
                    "retrieval_augmented_generation": {
                        "answer": generatedAnswer
                    }
                }
                return { openSearchResponse, query }
            }

            const startTime = new Date(Date.now());
            if(disableFilters) {
                await Promise.all([
                    hybridConversationSearch(newQuestion, newQuestion, parseFilters(newFilterContent, setErrorMessage), settings.activeConversation, settings.openSearchIndex, settings.embeddingModel, settings.modelName, settings.ragPassageCount)
                        .then(clean).then(populateChatFromOs),
                ]);
            }
            else {
                await Promise.all([
                    hybridSearchNoRag(newQuestion, parseFilters(newFilterContent, setErrorMessage), settings.openSearchIndex, settings.embeddingModel, true)
                        .then(anthropic_rag)
                        .then(clean_rag).then(populateChatFromOs),
                ]);
            }
            
        } finally {
            setLoadingMessage(null)
        }
    }


    return (
        <Card key={systemChat.id} padding="lg" radius="md" sx={{ 'borderStyle': 'none none solid none', 'borderColor': '#eee;', overflow: "visible" }} >
            <Group spacing="xs">
                {editing ? <Group p="0">
                    <ActionIcon size="xs" mr="0" >
                        <IconX onClick={(v) => {
                            setEditing(false);
                            setNewQuestion(systemChat.queryUsed)
                        }} />
                    </ActionIcon>
                </Group> :
                    <ActionIcon size="xs" mr="0" >
                        <IconEdit onClick={(v) => setEditing(true)} />
                    </ActionIcon>
                }
                {editing ? <TextInput variant="unstyled" w="90%" value={newQuestion} size="md" onKeyDown={handleInputKeyPress} onChange={handleInputChange}></TextInput> :
                    <Text size="md" fw={500} p="xs" pl="0">
                        {systemChat.queryUsed}
                    </Text>
                }

            </Group>

            {settings.auto_filter ? filters() : null}

            {editing ?
                <Group p="md">

                    <Button fz="xs" size="xs" color="teal" leftIcon={<IconPlayerPlayFilled size="0.6rem" />} onClick={(v) => {
                        rerunQuery()
                    }} >
                        Run
                    </Button>
                    <Button variant="light" color="teal" fz="xs" size="xs" onClick={() => {
                        setCurrentOsQuery(currentOsQuery)
                        setCurrentOsUrl(queryUrl)
                        openSearchQueryEditorOpenedHandlers.open()
                    }
                    }>
                        OpenSearch query editor
                    </Button>
                </Group>

                : null
            }

            <Text size="sm" sx={{ "whiteSpace": "pre-wrap" }} color={editing ? theme.colors.gray[4] : "black"} p="xs">
                {/* {textNodes} */}
                {replaceCitationsWithLinks(systemChat.response)}
                {systemChat.rawResults != null ?
                    <Container p="md">
                        <Title order={5}>OpenSearch results</Title>
                        <ScrollArea h="45rem">
                            <Prism language="markdown">{JSON.stringify(systemChat.rawResults, null, 4)}</Prism>
                        </ScrollArea>
                    </Container>
                    : null}
            </Text>
            <DocList documents={systemChat.hits} settings={settings} docsLoading={false}></DocList>
            <Stack p='xs' spacing='0'>
                {systemChat.originalQuery !== "" && systemChat.originalQuery !== systemChat.queryUsed ? 
                    <Text fz="xs" fs="italic" color="dimmed">
                            Original Query: {systemChat.originalQuery}
                    </Text>
                    :
                    null
                }
                <Text fz="xs" fs="italic" color="dimmed">
                    Interaction id: {systemChat.interaction_id ? systemChat.interaction_id : ""} 
                </Text>
            </Stack>
            

            <FeedbackButtons systemChat={systemChat} settings={settings} />
        </Card >
    );
}

function parseFilters(filterInputs: any, setErrorMessage: Dispatch<SetStateAction<string | null>>) {
    if (filterInputs == null) return null;
    let resultNeural: any = {
        "bool": {
            "filter": []
        }
    }
    let resultKeyword: any = {
        "bool": {
            "filter": []
        }
    }
    Object.entries(filterInputs).forEach(([filter, filterValue]) => {
        if (filter == null || filter == "") return;
        // ignore ntsb schema, handled separately below for auto filters
        if (filter == "location" || filter == "airplane_name" || filter == "date_start" || filter == "date_end" || filterValue == "unknown") {
            return
        }
        resultNeural["bool"]["filter"].push({
            "match": {
                [`properties.${filter}`]: filterValue
            }
        })
        resultKeyword["bool"]["filter"].push({
            "match": {
                [`properties.${filter}.keyword`]: filterValue
            }
        })
    });


    // for ntsb schema only
    if (filterInputs["location"] != null && filterInputs["location"] != "unknown") {
        resultKeyword["bool"]["filter"].push({
            "match": {
                "properties.entity.location": filterInputs["location"]
            }
        })
        resultNeural["bool"]["filter"].push({
            "match": {
                "properties.entity.location": filterInputs["location"]
            }
        })
    }
    if (filterInputs["airplane_name"] != null && filterInputs["airplane_name"] !== "unknown") {
        resultKeyword["bool"]["filter"].push({
            "match": {
                "properties.entity.aircraft": filterInputs["airplane_name"]
            }
        })
        resultNeural["bool"]["filter"].push({
            "match": {
                "properties.entity.aircraft": filterInputs["airplane_name"]
            }
        })
    }

    let range_query: any = {
        "range": {
            "properties.entity.day": {
            }
        }
    }
    if (filterInputs["date_start"] != null && filterInputs["date_start"] !== "unknown") {
        range_query.range["properties.entity.day"].gte = filterInputs["date_start"]
    }
    if (filterInputs["date_end"] != null && filterInputs["date_end"] !== "unknown") {
        range_query.range["properties.entity.day"].lte = filterInputs["date_end"]
    }
    if (range_query.range["properties.entity.day"].gte !== undefined
        || range_query.range["properties.entity.day"].lte !== undefined) {

        resultNeural.bool.filter.push(range_query)
        let keywordRange = {
            "range": {
                "properties.entity.day.keyword": {
                }
            }
        }
        keywordRange.range["properties.entity.day.keyword"] = range_query["range"]["properties.entity.day"]
        resultKeyword.bool.filter.push(keywordRange)
    }
    const result = {
        "keyword": resultKeyword,
        "neural": resultNeural
    }
    return result
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

const simplifyAnswer = async (question: string, answer: string) => {
    try {
        const response = await fetch('/aryn/simplify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: question,
                answer: answer
            })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        console.log("Simplify response is:", response)
        return response.text()
    } catch (error) {
        console.error('Error simplifying through proxy:', error);
        throw error;
    }
};

const anthropicRag = async (question: string, os_result: any) => {
    try {
        const response = await fetch('/aryn/anthropic_rag', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: question,
                os_result: os_result
            })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        console.log("AnthropicRAG response is:", response)
        return response.text()
    } catch (error) {
        console.error('Error in AnthropicRAG through proxy:', error);
        throw error;
    }
};

const interpretOsResult = async (question: string, os_result: string) => {
    try {
        const response = await fetch('/aryn/interpret_os_result', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: question,
                os_result: os_result
            })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        console.log("Simplify response is:", response)
        return response.text()
    } catch (error) {
        console.error('Error interpret_os_result through proxy:', error);
        throw error;
    }
};



export const ChatBox = ({ chatHistory, searchResults, setChatHistory, setSearchResults, streaming, setStreaming, setDocsLoading, setErrorMessage, settings, setSettings, refreshConversations, chatInputRef }:
    {
        chatHistory: (SystemChat)[], searchResults: SearchResultDocument[], setChatHistory: Dispatch<SetStateAction<any[]>>,
        setSearchResults: Dispatch<SetStateAction<any[]>>, streaming: boolean, setStreaming: Dispatch<SetStateAction<boolean>>,
        setDocsLoading: Dispatch<SetStateAction<boolean>>, setErrorMessage: Dispatch<SetStateAction<string | null>>, settings: Settings, setSettings: Dispatch<SetStateAction<Settings>>, refreshConversations: any,chatInputRef: any
    }) => {
    const theme = useMantineTheme();
    const [chatInput, setChatInput] = useState("");
    const [disableFilters, setDisableFilters] = useState(true);
    const [queryPlanner, setQueryPlanner] = useState(false);
    const [questionRewriting, setQuestionRewriting] = useState(false);
    const [filtersInput, setFiltersInput] = useState<{ [key: string]: string }>({});
    const [loadingMessage, setLoadingMessage] = useState<string | null>(null);
    const [currentOsQuery, setCurrentOsQuery] = useState<string>("");
    const [currentOsUrl, setCurrentOsUrl] = useState<string>("/opensearch/" + settings.openSearchIndex + "/_search?");
    const [openSearchQueryEditorOpened, openSearchQueryEditorOpenedHandlers] = useDisclosure(false);
    const {classes} = useStyles();
    const mobileScreen = useMediaQuery(`(max-width: ${theme.breakpoints.sm})`);
    const [containerWidth, setContainerWidth] = useState(null);
    const [settingsOpened, settingsHandler] = useDisclosure(false);
    const [filterError, setFilterError] = useState(false);
    const scrollAreaRef = useRef<HTMLDivElement>(null);


    useEffect(() => {
        setCurrentOsUrl("/opensearch/" + settings.openSearchIndex + "/_search?");
    }, [settings.openSearchIndex]);

    const scrollToBottom = () => {
        scrollAreaRef.current?.scrollTo({ top: scrollAreaRef.current.scrollHeight});
    };

    useEffect(() => {
        scrollToBottom();
    }, [chatHistory, streaming])

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
            let filterResponse;
            let filters: any;
            let filterContent: any = null;
            if (!disableFilters) {
                if (queryPlanner) {
                    filterResponse = await getFilters(chatInput, settings.modelName)
                    console.log(filterResponse)
                    if (filterResponse.ok) {
                        const filterData = await filterResponse.json();
                        const autoFilterRawResult = filterData.choices[0].message.content
                        if ((autoFilterRawResult.error !== undefined) &&
                            (autoFilterRawResult.error.type === 'timeout_exception')) {
                            const documents = new Array<SearchResultDocument>()
                            const chatResponse = "Timeout from OpenAI"
                            const interactionId = ""
                            setErrorMessage(chatResponse)
                            return null
                        }
                        try {
                            // let found = false
                            filterContent = JSON.parse(autoFilterRawResult);
                            filters = parseFilters(filterContent, setErrorMessage)
                        } catch (error) {
                            console.error('Error parsing JSON:', error);
                        }
                    }
                } else if (settings.required_filters.length > 0) {
                    filters = parseFilters(filtersInput, setErrorMessage)
                    filterContent = filtersInput
                    if (filters["keyword"]["bool"]["filter"].length != settings.required_filters.length) {
                        throw new Error("All required filters not populated");
                    }
                }
            } else {
                filters = null
            }
            console.log("Filters are: ", filters)
            let question: string = chatInput;
            let originalQuestion: string = question;
            if (questionRewriting) {
                setLoadingMessage("Rephrasing question with conversation context");
                const rephraseQuestionResponse = await rephraseQuestion(chatInput, chatHistoryInteractions, settings.modelName)
                const responseData = await rephraseQuestionResponse.json();
                const rephrasedQuestion = responseData.choices[0].message.content;
                console.log("Rephrased question to ", rephrasedQuestion)
                question = rephrasedQuestion
            }
            console.log("Question is: ", question)

            setLoadingMessage("Querying knowledge database with question: \"" + question + "\"");
            if (filters != null) {
                setLoadingMessage("Using filter: \"" + JSON.stringify(filters) + "\". Generating answer..");
            }

            const clean = async (result: any) => {
                const openSearchResponseAsync = result[0]
                const query = result[1]
                const openSearchResponse = await openSearchResponseAsync
                let generatedAnswer = openSearchResponse.ext.retrieval_augmented_generation.answer
                if (settings.simplify && openSearchResponse.hits.hits.length > 0) {
                    console.log("Simplifying answer: ", generatedAnswer)
                    setLoadingMessage("Simplifying answer..");
                    generatedAnswer = await simplifyAnswer(question, generatedAnswer)
                }
                await updateInteractionAnswer(openSearchResponse.ext.retrieval_augmented_generation.interaction_id, generatedAnswer, query)
                openSearchResponse.ext.retrieval_augmented_generation.answer = generatedAnswer
                return openSearchResponse
            }

            const clean_rag = async ({ openSearchResponse, query }: { openSearchResponse: any, query: any }) => {
                let generatedAnswer = openSearchResponse.ext.retrieval_augmented_generation.answer
                if (settings.simplify && openSearchResponse.hits.hits.length > 0) {
                    console.log("Simplifying answer: ", generatedAnswer)
                    generatedAnswer = await simplifyAnswer(question, generatedAnswer)
                }
                await updateInteractionAnswer(openSearchResponse.ext.retrieval_augmented_generation.interaction_id, generatedAnswer, query)
                openSearchResponse.ext.retrieval_augmented_generation.answer = generatedAnswer
                return { openSearchResponse, query }
            }

            const anthropic_rag = async (result: any) => {
                const openSearchResponseAsync = result[0]
                const query = result[1]
                const openSearchResponse = await openSearchResponseAsync
                let generatedAnswer = "Error"
                if (openSearchResponse.hits.hits.length > 0) {
                    console.log("Anthropic RAG time...")
                    generatedAnswer = await anthropicRag(question, openSearchResponse)
                }
                openSearchResponse["ext"] = {
                    "retrieval_augmented_generation": {
                        "answer": generatedAnswer
                    }
                }
                return { openSearchResponse, query }
            }


            const populateChatFromOs = (openSearchResults: any) => {
                console.log("Main processor ", openSearchResults)
                console.log("Main processor: OS results ", openSearchResults)
                const endTime = new Date(Date.now());
                const elpased = endTime.getTime() - startTime.getTime()
                console.log("Main processor: OS took seconds: ", elpased)
                const parsedOpenSearchResults = parseOpenSearchResults(openSearchResults, setErrorMessage)
                const newSystemChat = new SystemChat(
                    {
                        id: parsedOpenSearchResults.interactionId + "_system",
                        interaction_id: parsedOpenSearchResults.interactionId,
                        response: parsedOpenSearchResults.chatResponse,
                        ragPassageCount: settings.ragPassageCount,
                        modelName: settings.modelName,
                        queryUsed: question,
                        originalQuery: originalQuestion,
                        hits: parsedOpenSearchResults.documents,
                        filterContent: filterContent
                    });                    
                setChatHistory([...chatHistory, newSystemChat]);
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
            if(!settings.activeConversation) {
                const conversationId = await createConversation(chatInput);
                settings.activeConversation = conversationId.memory_id;
                setSettings(settings);
                refreshConversations();
            }

            if(disableFilters) {
                await Promise.all([
                    hybridConversationSearch(chatInput, question, filters, settings.activeConversation, settings.openSearchIndex, settings.embeddingModel, settings.modelName, settings.ragPassageCount)
                        .then(clean).then(populateChatFromOs),
                ]);
            }
            else {
                await Promise.all([
                    hybridSearchNoRag(question, parseFilters(filters, setErrorMessage), settings.openSearchIndex, settings.embeddingModel, true)
                        .then(anthropic_rag)
                        .then(clean_rag).then(populateChatFromOs), 
                ]);
            }
            
            
        } catch (e) {
            console.log(e)
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
                chatInputRef.current.disabled = false
            }
        }
    }

    // This method does all the search workflow execution
    const handleSubmit = async (e: React.FormEvent) => {
        if(chatInput.length === 0) {
            return;
        }
        if(!disableFilters && settings.required_filters.length > 0) {
            const someNonEmptyValues = Object.keys(filtersInput).length === 0 || Object.keys(filtersInput).some((key) => filtersInput[key] === '');
            if(someNonEmptyValues) {
                setFilterError(true);
                return;
            }
        }
        return handleSubmitParallelDocLoad(e)
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
        <>
            <OpenSearchQueryEditor
                openSearchQueryEditorOpened={openSearchQueryEditorOpened}
                openSearchQueryEditorOpenedHandlers={openSearchQueryEditorOpenedHandlers}
                currentOsQuery={currentOsQuery}
                currentOsUrl={currentOsUrl}
                setCurrentOsQuery={setCurrentOsQuery}
                setCurrentOsUrl={setCurrentOsUrl}
                setLoadingMessage={setLoadingMessage}
                chatHistory={chatHistory}
                setChatHistory={setChatHistory} />
            <ControlPanel settings={settings} setSettings={setSettings} controlPanelOpened={settingsOpened} onControlPanelClose={settingsHandler.close} />
            <Flex direction="column" pos='relative' className={classes.chatFlex}>
                <Stack className={classes.settingsStack} spacing='0'>
                   
                    <ActionIcon variant="transparent" className={classes.settingsIcon} onClick={settingsHandler.open}>
                        <IconSettings size="1.625rem" />
                    </ActionIcon>
                </Stack>
                {chatHistory.length === 0 && !loadingMessage && !streaming ? 
                            <Stack align='center' justify='center' className={classes.chatHistoryContainer} spacing='xs'>
                                <Image width='4em' src="./logo_only.png" />
                                <Text>How can I help you today?</Text>
                            </Stack>

                            :
                <ScrollArea className={classes.chatHistoryContainer} viewportRef={scrollAreaRef}>
                    <Container >
                        <Stack >
                            {chatHistory.map((chat, index) => {
                                return <SystemChatBox key={chat.id + "_system"} systemChat={chat} chatHistory={chatHistory} settings={settings} handleSubmit={handleSubmit}
                                    setChatHistory={setChatHistory} setSearchResults={setSearchResults} setErrorMessage={setErrorMessage}
                                    setLoadingMessage={setLoadingMessage} setCurrentOsQuery={setCurrentOsQuery} setCurrentOsUrl={setCurrentOsUrl} openSearchQueryEditorOpenedHandlers={openSearchQueryEditorOpenedHandlers} disableFilters={disableFilters}/>
                            }
                            )
                            }
                            {loadingMessage ? <LoadingChatBox loadingMessage={loadingMessage} /> : null}
                            
                            <Center>
                                {streaming ? <Loader size="xs" variant="dots" m="md" /> : ""}
                            </Center>
                        </Stack>
                       
                        
                    </Container>
                </ScrollArea>
                 }
                
                
            </Flex >
            <Container  className={classes.fixedBottomContainer}>
                <Group position={!disableFilters && settings.required_filters.length > 0 ? "apart" : 'left'} ml='auto' mr='auto' p='sm' h='3.5em' w={mobileScreen ? "90vw":"70vw"}>
                    <SearchControlPanel disableFilters={disableFilters} setDisableFilters={setDisableFilters} questionRewriting={questionRewriting} setQuestionRewriting={setQuestionRewriting}
                        queryPlanner={queryPlanner} setQueryPlanner={setQueryPlanner} chatHistory={chatHistory} setChatHistory={setChatHistory} openSearchQueryEditorOpenedHandlers={openSearchQueryEditorOpenedHandlers} settings={settings}></SearchControlPanel>
                    
                    {!disableFilters && settings.required_filters.length > 0 ? <FilterInput settings={settings} filtersInput={filtersInput} setFiltersInput={setFiltersInput} filterError={filterError} setFilterError={setFilterError} /> : null}
                </Group>
                <Center>
                    <TextInput

                        className={classes.inputBar}
                        onKeyDown={handleInputKeyPress}
                        onChange={handleInputChange}
                        ref={chatInputRef}
                        value={chatInput}
                        // icon={<IconSearch size="1.1rem" stroke={1.5} />}
                        radius="xl"
                        autoFocus
                        size='lg'
                        rightSectionWidth="auto"
                        rightSection={
                            <Group pr="0.2rem">
                                <Tooltip label="Rewrite question">
                                    <ActionIcon size={40} radius="xl" 
                                    sx={(theme) => ({
                                        color:  questionRewriting ? "white" :"#5688b0", 
                                        backgroundColor: questionRewriting ? "#5688b0" :"white",
                                        '&:hover': {
                                        backgroundColor: questionRewriting ? '#5688b0' : theme.colors.gray[2],
                                        },
                                    })}>
                                        <IconWriting size="1.3rem" stroke={1.5} onClick={() => setQuestionRewriting(o => !o)}  />
                                    </ActionIcon>
                                </Tooltip>
                                {mobileScreen &&
                                    <ActionIcon size={40} radius="xl" bg="#5688b0" variant="filled">
                                        <IconChevronRight size="1rem" stroke={2} onClick={handleSubmit} />
                                    </ActionIcon>
                                }  
                            </Group>
                            
                        }
                        placeholder="Ask me anything"
                        disabled={settings.activeConversation == null}
                    />
                </Center>
               
                    
            </Container>
        </>
    );
}
export const thumbToBool = (thumbValue: string) => {
    switch (thumbValue) {
        case "null": {
            return null;
        }
        case "up": {
            return true;
        }
        case "down": {
            return false;
        }
        default: {
            console.log("received unexpected feedback thumb value: " + thumbValue)
            return null;
        }
    }
}
