import React, { } from 'react';
import { ActionIcon, Alert, Card, Container, Flex, Group, LoadingOverlay, NavLink, Badge, ScrollArea, Stack, Text, Title, useMantineTheme, Tooltip, HoverCard, Anchor } from '@mantine/core';
import { useHover } from '@mantine/hooks';
import { IconInfoCircle, IconFileTypeHtml, IconLink, IconFileTypePdf } from '@tabler/icons-react';
import { SearchResultDocument, Settings } from './Types';

// const DocumentItem = ({ index, title, description, url, relevanceScore, properties }: SearchResultDocument) => {
const DocumentItem = ({ document }: { document: SearchResultDocument }) => {
    const theme = useMantineTheme();
    const { hovered, ref } = useHover();
    const openDocument = () => {
        if (document.isPdf()) {
            const dataString = JSON.stringify(document);
            console.log("You clicked: ", document)
            localStorage.setItem('pdfDocumentMetadata', dataString);
            window.open('/viewPdf');
        } else if (document.hasAbsoluteUrl()) {
            if(document.id === 'index-1') {
                // window.open(document.url + '#:~:text=' + 'Top%20Results,Phillip%20Griffith');
                const newWindow = window.open("about:blank"); // Open a blank tab first
                // newWindow?.focus();
                setTimeout(() => {
                    window.location.href = ( `${document.url}#:~:text=Top%20Results,Phillip%20Griffith`);
                }, 0)
            }
            else {
                window.open(document.url);
            }
        } else if (document.url.startsWith("/")) {
            window.open("/viewLocal" + document.url);
        } else {
            window.open(document.url);
        }
    }
    function icon() {
        if (document.isPdf()) {
            return (<IconFileTypePdf size="1rem" color={hovered ? theme.colors.blue[8] : theme.colors.blue[6]} />)
        } else if (document.url.endsWith("htm") || document.url.endsWith("html")) {
            return (<IconFileTypeHtml size="1rem" color={hovered ? theme.colors.gray[8] : theme.colors.gray[6]} />)
        }
        return (<IconLink size="1rem" />)
    };

    let snippet: String
    try {
        snippet = document.description.substring(0, 750)
    }
    catch (err) {
        snippet = ""
    }
    const parts: string[] = document.url.split("/");
    const filename: string = parts[parts.length - 1];
    if(!document.isPdf() && document.hasAbsoluteUrl()) {
        let redirectUrl = "";
        if(document.id === 'index-1') {
            redirectUrl = `${document.url}#:~:text=Top%20Results,Phillip%20Griffith`;
        }
        else if(document.id === 'index-0'){
            redirectUrl = `${document.url}#:~:text=Sort%20Benchmark%20Home%20Page,Engineering%20Dept`;
        }
        else if(document.id === 'index-3') {
            redirectUrl = `${document.url}#:~:text=GraySort,description%20document`;
        }
        else if(document.id === 'index-2') {
            redirectUrl = `${document.url}#:~:text=Common%20Rules,under%20clocking`;
        }
        else {
            redirectUrl = document.url;
        }
        return (
            <div ref={ref}>
                <HoverCard width="60%" shadow="md" position='bottom'>
                    <HoverCard.Target>
                        <Card mah="8rem" maw="20rem" w="auto" bg={hovered ? theme.colors.gray[1] : theme.colors.gray[0]} ml={theme.spacing.md} sx={{ cursor: 'pointer' }} shadow={hovered ? 'md' : 'none'} component="a" target="_blank"
                            mb="sm" href={redirectUrl}>
                            <Group p="xs" noWrap spacing="xs">
                                <Badge size="xs" color="gray" variant="filled" sx={{ overflow: "visible" }} > {document.index}</Badge>
                                <Text size="sm" c={hovered ? theme.colors.blue[8] : theme.colors.dark[8]} truncate>
                                    {document.title != "Untitled" ? document.title :
                                        (document.properties.entity ? document.properties.entity.accidentNumber ?? "Untitled": "Untitled")}
                                </Text>
                            </Group>
                            <Group p="xs" noWrap spacing="xs">
                                {icon()}
                                <Text size="xs" color="gray.7" truncate>{filename}</Text>
                            </Group>
                        </Card >
                    </HoverCard.Target>
                    <HoverCard.Dropdown>
                        <ScrollArea h='20vh'>
                            <Title order={5}>{document.title}</Title>
                            <Anchor href={document.url} target="_blank">
                                <Text fz="xs">{document.url} </Text>
                            </Anchor>
                            <Group>
                                {
                                    "entity" in document.properties ? ["location", "aircraftType", "day"].map(key => {
                                        if (document.properties.entity[key] != "None") return (
                                            <Badge key={key} size="xs" variant="filled">
                                                {key} {document.properties.entity[key]}
                                            </Badge>
                                        )
                                    }) : null
                                }
                            </Group>
                            <Text> {document.description}</Text>
                        </ScrollArea>
                    </HoverCard.Dropdown>
                </HoverCard>
            </div >
        );
    }

    return (
        <div ref={ref}>
            <HoverCard width="60%" shadow="md" position='bottom'>
                <HoverCard.Target>
                    <Card mah="8rem" maw="20rem" w="auto" bg={hovered ? theme.colors.gray[1] : theme.colors.gray[0]} ml={theme.spacing.md} sx={{ cursor: 'pointer' }} shadow={hovered ? 'md' : 'none'} component="a" onClick={() => { openDocument() }} target="_blank"
                        mb="sm">
                        <Group p="xs" noWrap spacing="xs">
                            <Badge size="xs" color="gray" variant="filled" sx={{ overflow: "visible" }} > {document.index}</Badge>
                            <Text size="sm" c={hovered ? theme.colors.blue[8] : theme.colors.dark[8]} truncate>
                                {document.title != "Untitled" ? document.title :
                                    (document.properties.entity ? document.properties.entity.accidentNumber ?? "Untitled": "Untitled")}
                            </Text>
                        </Group>
                        <Group p="xs" noWrap spacing="xs">
                            {icon()}
                            <Text size="xs" color="gray.7" truncate>{filename}</Text>
                        </Group>
                    </Card >
                </HoverCard.Target>
                <HoverCard.Dropdown>
                    <ScrollArea h='20vh'>
                        <Title order={5}>{document.title}</Title>
                        <Anchor href={document.url} target="_blank">
                            <Text fz="xs">{document.url} </Text>
                        </Anchor>
                        <Group>
                            {
                                "entity" in document.properties ? ["location", "aircraftType", "day"].map(key => {
                                    if (document.properties.entity[key] != "None") return (
                                        <Badge key={key} size="xs" variant="filled">
                                            {key} {document.properties.entity[key]}
                                        </Badge>
                                    )
                                }) : null
                            }
                        </Group>
                        <Text> {document.description}</Text>
                    </ScrollArea>
                </HoverCard.Dropdown>
            </HoverCard>
        </div >
    );
}

export const DocList = ({ documents, settings, docsLoading }: { documents: SearchResultDocument[], settings: Settings, docsLoading: boolean }) => {
    const theme = useMantineTheme();
    const icon = <IconInfoCircle />;

    return (
        <Container bg="white">
            <ScrollArea w="90%" sx={{ overflow: "visible" }}>
                <Container sx={{ overflow: "visible" }}>
                    <LoadingOverlay visible={docsLoading} overlayBlur={2} />
                    <Group noWrap>
                        {documents.map(document => (
                            < DocumentItem
                                key={document.id + Math.random()}
                                document={document}
                            />
                        )
                        )
                        }
                    </Group>
                </Container>
            </ScrollArea>
        </Container >
    );
}
