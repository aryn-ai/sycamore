import React, { Dispatch, SetStateAction, useState, useEffect, useRef } from 'react';
import { ActionIcon, createStyles, Loader, Navbar, Text, useMantineTheme, rem, Center, Container, Group, Anchor, TextInput } from '@mantine/core';
import { Settings } from './Types'
import { createConversation, deleteConversation, getConversations } from './OpenSearch';
import { IconChevronRight, IconMessagePlus, IconTrash } from '@tabler/icons-react';
import { useHover } from '@mantine/hooks';
const useStyles = createStyles((theme) => ({
    wrapper: {
        display: 'flex',
    },
    main: {
        flex: 1,
        backgroundColor: theme.colorScheme === 'dark' ? theme.colors.dark[6] : theme.colors.gray[0],
    },
    title: {
        textTransform: 'uppercase',
        letterSpacing: rem(-0.25),
    },
    link: {
        boxSizing: 'border-box',
        display: 'block',
        textDecoration: 'none',
        borderRadius: theme.radius.md,
        color: theme.colorScheme === 'dark' ? theme.colors.dark[0] : theme.colors.gray[7],
        fontSize: theme.fontSizes.sm,
        fontWeight: 500,
        height: rem(44),
        lineHeight: rem(44),

        '&:hover': {
            backgroundColor: theme.colorScheme === 'dark' ? theme.colors.dark[5] : theme.colors.gray[1],
            color: theme.colorScheme === 'dark' ? theme.white : theme.black,
        },
    },

    linkActive: {
        '&, &:hover': {
            borderLeftColor: theme.fn.variant({ variant: 'filled', color: theme.primaryColor })
                .background,
            backgroundColor: theme.fn.variant({ variant: 'filled', color: theme.primaryColor })
                .background,
            color: theme.white,
        },
    },
}))
export function setActiveConversation(conversationId: string, settings: Settings, setSettings: Dispatch<SetStateAction<Settings>>, loadActiveConversation: any) {
    settings.activeConversation = conversationId
    setSettings(settings)
    loadActiveConversation(conversationId)
}
const NavBarConversationItem = ({ conversation, conversations, setConversations, selectConversation, loading, settings }: { conversation: any, conversations: any[], setConversations: any, selectConversation: any, loading: any, settings: any }) => {
    const { classes, cx } = useStyles();
    return (
        <Group key={conversation.id + "_navbar_row"} id={conversation.id + "_navbar_row"} >
            <ActionIcon size="1rem" ml="sm" mr="xs" component="button"
                onClick={(event) => {
                    console.log("Removing ", conversation.id)
                    deleteConversation(conversation.id)
                    const newConversations = conversations.filter((c) => c.id !== conversation.id);
                    console.log("newLinks", newConversations)
                    setConversations(newConversations);
                }}>
                <IconTrash size="1.125rem" />
            </ActionIcon>
            <Anchor
                className={cx(classes.link, { [classes.linkActive]: conversation.id === settings.activeConversation })}
                w="12rem"
                pl="xs"
                ml="xs"
                key={conversation.id}
                onClick={(event) => {
                    event.preventDefault();
                    selectConversation(conversation.id)
                }}
            >
                <span>{conversation.name}</span>
                {loading ? <Loader size="xs" variant="dots" /> : ""}
            </Anchor>
        </Group>
    );
}

const NewConversationInput = ({ refreshConversations, setErrorMessage }: { refreshConversations: any, setErrorMessage: Dispatch<SetStateAction<string | null>> }) => {
    const [newConversationName, setNewConversationName] = useState("")
    const newConversationInputRef = useRef(null);
    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setNewConversationName(e.target.value);
    };
    async function handleSubmit() {
        try {
            const createConversationResponse = await createConversation(newConversationName)
            const conversationId = await createConversationResponse;
            refreshConversations()
        } catch (e) {
            console.log("Error creating conversation: ", e)
            if (typeof e === "string") {
                setErrorMessage(e.toUpperCase())
            } else if (e instanceof Error) {
                setErrorMessage(e.message)
            }
        } finally {
            setNewConversationName("")
        }
    }
    const handleInputKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            handleSubmit();
        }
    };
    return (
        <Group>
            <form className="input-form">
                <TextInput
                    onKeyDown={handleInputKeyPress}
                    onChange={handleInputChange}
                    ref={newConversationInputRef}
                    value={newConversationName}
                    w="17rem"
                    radius="sm"
                    fz="xs"
                    rightSection={
                        <ActionIcon size={32} radius="sm">
                            <IconMessagePlus size="1rem" stroke={2} onClick={handleSubmit} />
                        </ActionIcon>
                    }
                    placeholder="New conversation"
                />
            </form>
        </Group >
    )
}

export const ConversationListNavbar = ({ navBarOpened, settings, setSettings, setErrorMessage, loadingConversation, loadActiveConversation }:
    { navBarOpened: boolean, settings: Settings, setSettings: Dispatch<SetStateAction<Settings>>, setErrorMessage: Dispatch<SetStateAction<string | null>>, loadingConversation: boolean, loadActiveConversation: any }) => {
    const theme = useMantineTheme();
    const { classes, cx } = useStyles();
    const [loading, setLoading] = useState(false);
    const [conversations, setConversations] = useState<any>([]);

    const selectConversation = (conversationId: string) => {
        console.info("Set active conversation to ", conversationId)
        setActiveConversation(conversationId, settings, setSettings, loadActiveConversation);
        refreshConversations();
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

    useEffect(() => {
        setLoading(true)
        refreshConversations()
        setLoading(false)
    }, []);
    return (
        <Navbar
            width={{ sm: navBarOpened ? "20rem" : 0 }}
            sx={{
                overflow: "hidden",
                transition: "width 150ms ease, min-width 150ms ease",
                backgroundColor: theme.colorScheme === 'dark' ? theme.colors.dark[6] : theme.colors.gray[0]
            }}>
            <Navbar.Section p="md" sx={{ 'borderStyle': 'none none solid none', 'borderColor': '#eee;' }}>
                <Text weight={500} size="sm" className={classes.title} color="dimmed">
                    <Group>
                        <Text>Conversations</Text>
                        <Container m="0">{loadingConversation ? <Center p="md"><Loader size="xs" variant="dots" /></Center> : null}</Container>
                    </Group>
                </Text>
            </Navbar.Section>
            {loading ? <Center p="md"><Loader size="xs" /></Center> : null}
            <Navbar.Section p="xs" m="xs">
                <NewConversationInput refreshConversations={refreshConversations} setErrorMessage={setErrorMessage} />
            </Navbar.Section>
            <Navbar.Section grow className={classes.wrapper}>
                <div className={classes.main}>
                    {
                        conversations.map((conversation: any) => (
                            <NavBarConversationItem key={conversation.id} conversation={conversation} conversations={conversations} setConversations={setConversations} selectConversation={selectConversation} loading={loading} settings={settings} />
                        ))
                    }
                </div>
            </Navbar.Section>
        </Navbar>
    );
}
