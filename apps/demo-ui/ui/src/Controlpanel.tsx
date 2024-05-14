import React, { Dispatch, SetStateAction, useEffect, useState } from 'react';
import { Button, Divider, Group, Modal, NativeSelect, ScrollArea, Stack, Text, createStyles, useMantineTheme } from '@mantine/core';
import { IconRefresh } from '@tabler/icons-react';
import { Settings } from './Types'
import { getIndices, getEmbeddingModels, FEEDBACK_INDEX_NAME, createFeedbackIndex } from './OpenSearch';

export const ControlPanel = ({ settings, setSettings, controlPanelOpened, onControlPanelClose }: { settings: Settings, setSettings: Dispatch<SetStateAction<Settings>>, controlPanelOpened: boolean, onControlPanelClose: any }) => {
    const [availableIndices, setAvailableIndices] = useState(new Array<string>())
    const [availableEmbeddings, setAvailableEmbeddings] = useState(new Array<string>())
    const newsettings = settings

    const getIndicesAndEmbeddings = async () => {
        const [getIndicesResponse, getEmbeddingsResponse] = await Promise.all([
            getIndices(),
            getEmbeddingModels(),
        ])
        const newIndiciesMaybeWFeedback = Object.keys(getIndicesResponse).filter((key) => !key.startsWith("."))
        var newIndicies;
        if(newIndiciesMaybeWFeedback.includes(FEEDBACK_INDEX_NAME)) {
            newIndicies = newIndiciesMaybeWFeedback.filter((name) => name !== FEEDBACK_INDEX_NAME)
        } else {
            newIndicies = newIndiciesMaybeWFeedback
            createFeedbackIndex()
        }

        const hits = getEmbeddingsResponse.hits.hits
        const models = []
        for (const idx in hits) {
            models.push(hits[idx]._id)
        }

        return [newIndicies, models]
    }

    useEffect(() => {
        const doit = async () => {
            const [indexNames, modelIds] = await getIndicesAndEmbeddings()
            newsettings.openSearchIndex = indexNames[0];
            newsettings.embeddingModel = modelIds[0];
            setSettings(newsettings)
            setAvailableIndices(indexNames)
            setAvailableEmbeddings(modelIds)
        }
        doit()
    }, []);

    return (
        <Modal opened={controlPanelOpened} onClose={onControlPanelClose} title="Options" centered>
        < Stack> 
            
            <Divider orientation="horizontal" />
            <Group position='apart'>
            <Text fz="xs">RAG passage count:</Text>
            <NativeSelect
                data={["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]}
                defaultValue={settings.ragPassageCount}
                onChange={(event) => {
                    const newsettings = settings
                    newsettings.ragPassageCount = +event.currentTarget.value
                    setSettings(newsettings)
                }
                }
            />
            </Group>
            <Group position='apart'>
            <Text fz="xs">AI model:</Text>
            <NativeSelect
                data={settings.availableModels}
                defaultValue={settings.modelName}
                onChange={(event) => {
                    const newsettings = settings
                    newsettings.modelName = event.currentTarget.value
                    setSettings(newsettings)
                }
                }
            />
            </Group>
            <Group position='apart'>
            <Text fz="xs">OpenSearch index:</Text>
            <NativeSelect
                data={Array.from(availableIndices)}
                defaultValue={settings.openSearchIndex}
                onChange={(event) => {
                    const newsettings = settings
                    newsettings.openSearchIndex = event.currentTarget.value
                    setSettings(newsettings)
                }
                }
            />
            </Group>
            <Group position='apart'>
            <Text fz="xs">Embedding Model:</Text>
            <NativeSelect
                data={Array.from(availableEmbeddings)}
                defaultValue={settings.embeddingModel}
                onChange={(event) => {
                    const newsettings = settings
                    newsettings.embeddingModel = event.currentTarget.value
                    setSettings(newsettings)
                }
                }
            />
            </Group>
         </Stack > 
         </Modal>
    );
}
