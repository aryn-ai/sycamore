import React, { Dispatch, SetStateAction, useState, useEffect, useRef } from 'react';
import { ActionIcon, createStyles, Loader, Navbar, Text, useMantineTheme, rem, Center, Container, Group, Anchor, TextInput, Button, Image, em, MediaQuery, NavLink, Menu } from '@mantine/core';
import { Settings, SystemChat } from './Types'
import { createConversation, deleteConversation } from './OpenSearch';
import { IconMessagePlus, IconTrash, IconMessage, IconDotsVertical } from '@tabler/icons-react';
import { useMediaQuery } from '@mantine/hooks';

const useStyles = createStyles((theme) => ({
    wrapper: {
        display: 'flex',
        padding: "1rem"
    },
    main: {
        flex: 1,
        // padding: "1rem",
        width: '100%',
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
        // width: "80%",
        // maxWidth: 'calc(100% - 40px)', 
        overflow: 'hidden', 
        textOverflow: 'ellipsis',

        [theme.fn.smallerThan('sm')]: {
            fontSize: theme.fontSizes.lg,
            paddingLeft: rem(15),
            borderRadius: theme.radius.xl
        },
        
        '&:hover': {
            backgroundColor: theme.colorScheme === 'dark' ? theme.colors.dark[5] : theme.colors.gray[2],
            color: theme.colorScheme === 'dark' ? theme.white : theme.black,
        },

    },
    linkRow: {
        width: "100%",
        marginTop: rem(10),
        marginBottom: rem(10),
    },

    linkActive: {
        '&, &:hover': {
            borderLeftColor: theme.fn.variant({ variant: 'filled', color: theme.primaryColor })
                .background,
            backgroundColor: '#5688b0',
            color: theme.white,
        },
    },
    conversationName: {
        [theme.fn.smallerThan('sm')]: {
            width: '55vw'
        },
        [theme.fn.largerThan('sm')]: {
            width: 'calc(175px - 7rem)'
        },
        [theme.fn.largerThan('lg')]: {
            width: 'calc(275px - 7rem)'
        },
    }
}))
export function setActiveConversation(conversationId: string, settings: Settings, setSettings: Dispatch<SetStateAction<Settings>>, loadActiveConversation: any) {
    settings.activeConversation = conversationId
    setSettings(settings)
    loadActiveConversation(conversationId)
}
const NavBarConversationItem = ({ conversation, conversations, setConversations, selectConversation, loading, settings, setSettings, setChatHistory, setNavBarOpened, openErrorDialog }: { conversation: any, conversations: any[], setConversations: any, selectConversation: any, loading: any, settings: any, setChatHistory: any, setSettings: Dispatch<SetStateAction<Settings>>, setNavBarOpened: any, openErrorDialog: any }) => {
    const { classes, cx } = useStyles();
    
    const handleDelete = (event: React.MouseEvent) => {
        try {
            event.stopPropagation();
            console.log("Removing ", conversation.id)
            deleteConversation(conversation.id)
            if(conversation.id === settings.activeConversation) {
                setChatHistory(new Array<SystemChat>());
                settings.activeConversation = "";
                setSettings(new Settings(settings));
            }
            const newConversations = conversations.filter((c) => c.id !== conversation.id);
            
            console.log("newLinks", newConversations)
            setConversations(newConversations);
        } catch(error: any) {
            openErrorDialog("Error deleting conversation: " + error.message);
            console.error("Error deleting conversation: " + error.message);
        }
    }
    return (
        <>
        <Group key={conversation.id + "_navbar_row"} id={conversation.id + "_navbar_row"} noWrap className={classes.linkRow}>
            <Anchor
                className={cx(classes.link, { [classes.linkActive]: conversation.id === settings.activeConversation })}
                key={conversation.id}
                onClick={(event) => {
                    event.preventDefault();
                    selectConversation(conversation.id)
                    setNavBarOpened(false);
                }}
                w="100%"
            >
                <Group position='apart' align='center' noWrap>
                    <Group position='left' noWrap pl='xs'>
                        <IconMessage />
                        <Text className={classes.conversationName}  truncate >{conversation.name}</Text>
                    </Group>
                    <Menu shadow="md" position="bottom">
                        <Menu.Target>
                            <ActionIcon c='white' onClick={(e) => e.stopPropagation()}  sx={(theme) => ({
                                borderRadius: 100,
                                '&:hover': {
                                backgroundColor:  conversation.id === settings.activeConversation ? '#4f779b' : theme.colors.gray[4],
                                },
                            })}>
                                <IconDotsVertical color={conversation.id === settings.activeConversation ? "white" : "gray"}/>
                            </ActionIcon>
                        </Menu.Target>
                        <Menu.Dropdown p={0} m={0}>
                            <Menu.Item icon={<IconTrash size={14}/>} onClick={handleDelete}>Delete</Menu.Item>
                        </Menu.Dropdown>
                    </Menu>
                </Group>
                 
                {loading ? <Loader size="xs" variant="dots" /> : ""}
            </Anchor>
        </Group>

        </>
    );
}

const NewConversationInput = ({ refreshConversations, setErrorMessage, chatInputRef, settings, setSettings, setChatHistory, setNavBarOpened, loadActiveConversation, navBarOpened }:
     { refreshConversations: any, setErrorMessage: Dispatch<SetStateAction<string | null>>, chatInputRef: any, settings: Settings, setSettings: Dispatch<SetStateAction<Settings>>, setChatHistory: Dispatch<SetStateAction<Array<SystemChat>>>, setNavBarOpened: any, loadActiveConversation: any, navBarOpened: boolean
 }) => {
    const [newConversationName, setNewConversationName] = useState("")
    const [error, setError] = useState(false);
    const newConversationInputRef = useRef(null);
    const theme = useMantineTheme();
    const mobileScreen = useMediaQuery(`(max-width: ${theme.breakpoints.sm})`);
    
    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setError(false);
        setNewConversationName(e.target.value);
    };
    async function handleSubmit() {
        try {
            if(newConversationName === "") {
                setError(true);
                return;
            }
            setError(false);
            const createConversationResponse = await createConversation(newConversationName)
            const conversationId = createConversationResponse.memory_id;
            setActiveConversation(conversationId, settings, setSettings, loadActiveConversation);
            setNavBarOpened(false);
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
    const handleBlur = () => {
        if(error) {
            setError(false);
        }
    }
    
    return (
        <> 
            <TextInput
                onKeyDown={handleInputKeyPress}
                onChange={handleInputChange}
                ref={newConversationInputRef}
                value={newConversationName}
                w='100%'
                radius="sm"
                fz="xs"
                
                size={mobileScreen ? "md" : "sm"}
                rightSection={ mobileScreen ?
                    <ActionIcon size={32} radius="sm" c={error ? 'red': '#5688b0'}>
                        <IconMessagePlus size="1rem" stroke={2} onClick={handleSubmit} />
                    </ActionIcon>
                    :
                    ""
                }
                placeholder="New conversation"
                error={error ? "Conversation name cannot be empty" : ""}
                onBlur={handleBlur}
            />
        </>
    )
}


export const ConversationListNavbar = ({ navBarOpened, settings, setSettings, setErrorMessage, loadingConversation, loadActiveConversation, conversations, refreshConversations, setConversations, setChatHistory, chatInputRef, setNavBarOpened, openErrorDialog
}:
    { navBarOpened: boolean, settings: Settings, setSettings: Dispatch<SetStateAction<Settings>>, setErrorMessage: Dispatch<SetStateAction<string | null>>, loadingConversation: boolean, loadActiveConversation: any, conversations: any, refreshConversations: any, setConversations: any, setChatHistory: any, chatInputRef: any ,setNavBarOpened:any, openErrorDialog: any
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
            width={{sm: 200, lg: 300}}
            sx={{
                overflow: "hidden",
                transition: "width 150ms ease, min-width 150ms ease",
                backgroundColor: theme.colorScheme === 'dark' ? theme.colors.dark[6] : theme.colors.gray[0]
            }}>
            {loading ? <Center p="md"><Loader size="xs" /></Center> : null}
            <Navbar.Section p="xs" m="xs" sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <NewConversationInput refreshConversations={refreshConversations} setErrorMessage={setErrorMessage} chatInputRef={chatInputRef} settings={settings} setSettings={setSettings} setChatHistory={setChatHistory} setNavBarOpened={setNavBarOpened} loadActiveConversation={loadActiveConversation} navBarOpened={navBarOpened}/>
            </Navbar.Section>
            <Navbar.Section grow className={classes.wrapper}  w={{sm: 200, lg: 300}}>
            {loadingConversation ? (
            <Container m="0"> 
                <Center p="md">
                    <Loader size="sm" variant="dots" />
                </Center>
            </Container>
            ) : (
            <div className={classes.main} >
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
}