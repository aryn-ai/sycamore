import React, { Dispatch, SetStateAction, useState, useEffect, useRef } from 'react';
import { ActionIcon, createStyles, Loader, Navbar, Text, useMantineTheme, rem, Center, Container, Group, Anchor, TextInput, Button, Image, em, MediaQuery } from '@mantine/core';
import { Settings, SystemChat } from './Types'
import { createConversation, deleteConversation, getConversations } from './OpenSearch';
import { IconChevronRight, IconChevronLeft, IconMessagePlus, IconTrash, IconPlus } from '@tabler/icons-react';
import { useHover } from '@mantine/hooks';
import { useMediaQuery } from '@mantine/hooks';

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
        display: 'inline-block',
        textDecoration: 'none',
        borderRadius: theme.radius.md,
        color: theme.colorScheme === 'dark' ? theme.colors.dark[0] : theme.colors.gray[7],
        fontSize: theme.fontSizes.sm,
        fontWeight: 500,
        padding: rem(5),
        width: "80%",
        maxWidth: 'calc(100% - 40px)', 
        overflow: 'hidden', 
        textOverflow: 'ellipsis',

        [theme.fn.smallerThan('md')]: {
            fontSize: theme.fontSizes.lg,
            paddingLeft: rem(15),
            borderRadius: theme.radius.xl
        },
        
        '&:hover': {
            backgroundColor: theme.colorScheme === 'dark' ? theme.colors.dark[5] : theme.colors.gray[1],
            color: theme.colorScheme === 'dark' ? theme.white : theme.black,
        },

    },
    linkRow: {
        [theme.fn.smallerThan('md')]: {
            marginTop: rem(5),
            marginBottom: rem(5)
        },
      
    },

    linkActive: {
        '&, &:hover': {
            borderLeftColor: theme.fn.variant({ variant: 'filled', color: theme.primaryColor })
                .background,
            backgroundColor: '#5688b0',
            color: theme.white,
            underline: "hover"
        },
    },
}))
export function setActiveConversation(conversationId: string, settings: Settings, setSettings: Dispatch<SetStateAction<Settings>>, loadActiveConversation: any) {
    settings.activeConversation = conversationId
    setSettings(settings)
    loadActiveConversation(conversationId)
}
const NavBarConversationItem = ({ conversation, conversations, setConversations, selectConversation, loading, settings, setSettings, setChatHistory, setNavBarOpened }: { conversation: any, conversations: any[], setConversations: any, selectConversation: any, loading: any, settings: any, setChatHistory: any, setSettings: any, setNavBarOpened: any }) => {
    const { classes, cx } = useStyles();
    const theme = useMantineTheme();
    const mobileScreen = useMediaQuery(`(max-width: ${theme.breakpoints.md})`);
    return (
        <Group key={conversation.id + "_navbar_row"} id={conversation.id + "_navbar_row"} noWrap className={classes.linkRow}>
            <ActionIcon size={"md"} ml="sm"  component="button"
                c={'#5688b0'}
                onClick={(event) => {
                    console.log("Removing ", conversation.id)
                    deleteConversation(conversation.id)
                    if(conversation.id === settings.activeConversation) {
                        setChatHistory(new Array<SystemChat>());
                        settings.activeConversation = "";
                        setSettings(settings);
                    }
                    const newConversations = conversations.filter((c) => c.id !== conversation.id);
                    
                    console.log("newLinks", newConversations)
                    setConversations(newConversations);
                }}>
                <IconTrash size={mobileScreen ? "1.5rem" : "1.125rem"} />
            </ActionIcon>
            <Anchor
                className={cx(classes.link, { [classes.linkActive]: conversation.id === settings.activeConversation })}
                key={conversation.id}
                onClick={(event) => {
                    event.preventDefault();
                    selectConversation(conversation.id)
                    setNavBarOpened(false);
                }}
            >
                <span>{conversation.name}</span>
                {loading ? <Loader size="xs" variant="dots" /> : ""}
            </Anchor>
        </Group>
    );
}

const NewConversationInput = ({ refreshConversations, setErrorMessage, chatInputRef, settings, setSettings, setChatHistory, 
    setNavBarOpened 
}: { refreshConversations: any, setErrorMessage: Dispatch<SetStateAction<string | null>>, chatInputRef: any, settings: Settings, setSettings: Dispatch<SetStateAction<Settings>>, setChatHistory: Dispatch<SetStateAction<Array<SystemChat>>>,
    setNavBarOpened: any
 }) => {
    const theme = useMantineTheme();
    const mobileScreen = useMediaQuery(`(max-width: ${theme.breakpoints.md})`);
    async function handleSubmit() {
        try {
            if (chatInputRef.current !== null) {
                chatInputRef.current.focus();
                if(settings.activeConversation !== "") {
                    setChatHistory(new Array<SystemChat>());
                    settings.activeConversation = "";
                    setSettings(settings);
                }
                setNavBarOpened(false);
            }
        } catch (e) {
            console.log("Error creating conversation: ", e)
            if (typeof e === "string") {
                setErrorMessage(e.toUpperCase())
            } else if (e instanceof Error) {
                setErrorMessage(e.message)
            }
        }
    }
    
    return (
        <>
            <Button 
                variant="outline" 
                radius="xl" 
                size={mobileScreen ? 'md': 'sm'}
                c={'#5688b0'}
                fullWidth
                leftIcon={
                    <ActionIcon size={32} radius="sm" c={'#5688b0'}>
                        <IconPlus size="1rem" stroke={3} color={theme.colors.blue[6]}  />
                    </ActionIcon>
                } 
                onClick={handleSubmit} 
            >
                New Conversation
            </Button> 
        </>
    )
}


export const ConversationListNavbar = ({ navBarOpened, settings, setSettings, setErrorMessage, loadingConversation, loadActiveConversation, conversations, refreshConversations, setConversations, setChatHistory, chatInputRef, 
    setNavBarOpened
}:
    { navBarOpened: boolean, settings: Settings, setSettings: Dispatch<SetStateAction<Settings>>, setErrorMessage: Dispatch<SetStateAction<string | null>>, loadingConversation: boolean, loadActiveConversation: any, conversations: any, refreshConversations: any, setConversations: any, setChatHistory: any, chatInputRef: any
        ,setNavBarOpened:any
     }) => {
    const theme = useMantineTheme();
    const { classes, cx } = useStyles();
    const [loading, setLoading] = useState(false);

    const selectConversation = (conversationId: string) => {
        console.info("Set active conversation to ", conversationId)
        setActiveConversation(conversationId, settings, setSettings, loadActiveConversation);
        refreshConversations();
    }

    useEffect(() => {
        setLoading(true)
        refreshConversations()
        setLoading(false)
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
                backgroundColor: theme.colorScheme === 'dark' ? theme.colors.dark[6] : theme.colors.gray[0]
            }}>
            {loading ? <Center p="md"><Loader size="xs" /></Center> : null}
            <Navbar.Section p="xs" m="xs" sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <NewConversationInput refreshConversations={refreshConversations} setErrorMessage={setErrorMessage} chatInputRef={chatInputRef} settings={settings} setSettings={setSettings} setChatHistory={setChatHistory} setNavBarOpened={setNavBarOpened}/>
            </Navbar.Section>
            <Navbar.Section grow className={classes.wrapper}>
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
                />
                ))}
            </div>
            )}
            </Navbar.Section>
        </Navbar>
    );
}
