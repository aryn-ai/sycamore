import React, { Dispatch, FormEvent, SetStateAction, useEffect, useState } from 'react';
import { Button, Divider, Group, Modal, NativeSelect, ScrollArea, Stack, Text, createStyles, useMantineTheme } from '@mantine/core';
import { IconRefresh } from '@tabler/icons-react';
import { Settings } from './Types'
import { getIndices, getEmbeddingModels, FEEDBACK_INDEX_NAME, createFeedbackIndex } from './OpenSearch';

export const ControlPanel = ({ settings, setSettings, controlPanelOpened, onControlPanelClose, openErrorDialog }: { settings: Settings, setSettings: Dispatch<SetStateAction<Settings>>, controlPanelOpened: boolean, onControlPanelClose: any, openErrorDialog:any }) => {
    const [availableIndices, setAvailableIndices] = useState(new Array<string>())
    const [availableEmbeddings, setAvailableEmbeddings] = useState(new Array<string>())
    const [formValues, setFormValues] = useState({
        ragPassageCount: settings.ragPassageCount,
        modelName: settings.modelName,
        openSearchIndex: settings.openSearchIndex,
        embeddingModel: settings.embeddingModel,
    });
    

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
            try {
                const [indexNames, modelIds] = await getIndicesAndEmbeddings()
                setSettings(settings => ({
                    ...settings,
                    openSearchIndex: indexNames[0],
                    embeddingModel: modelIds[0],
                }));
                setFormValues(prev => ({
                    ...prev,
                    openSearchIndex: indexNames[0],
                    embeddingModel: modelIds[0],
                }))
                setAvailableIndices(indexNames)
                setAvailableEmbeddings(modelIds)
            }
            catch(error: any) {
                openErrorDialog("Error loading settings: " + error.message);
                console.error("Error loading settings:", error);
            }
        }
        doit()
    }, []);

    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        setSettings(prevSettings => ({ ...prevSettings, ...formValues })); 
        setTimeout(() => {
            console.log(settings);
            
        })
        onControlPanelClose();
    }

    return (
        <Modal opened={controlPanelOpened} onClose={onControlPanelClose} title="Options" centered>
            <form onSubmit={handleSubmit}>
                <Stack> 
                    
                    <Divider orientation="horizontal" />
                    <Group position='apart'>
                    <Text fz="xs">RAG passage count:</Text>
                    <NativeSelect
                        data={["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]}
                        defaultValue={formValues.ragPassageCount}
                        onChange={(event) => {
                            setFormValues(prev => ({
                                ...prev,
                                ragPassageCount: +event.currentTarget.value
                            }));
                        }
                        }
                    />
                    </Group>
                    <Group position='apart'>
                    <Text fz="xs">AI model:</Text>
                    <NativeSelect
                        data={settings.availableModels}
                        defaultValue={formValues.modelName}
                        onChange={(event) => {
                            setFormValues(prev => ({
                                ...prev,    
                                modelName: event.currentTarget.value, 
                            }));
                        }
                        }
                    />
                    </Group>
                    <Group position='apart'>
                    <Text fz="xs">OpenSearch index:</Text>
                    <NativeSelect
                        data={Array.from(availableIndices)}
                        defaultValue={formValues.openSearchIndex}
                        onChange={(event) => {
                            setFormValues(prev => ({
                                ...prev,    
                                openSearchIndex: event.currentTarget.value, 
                            }));
                        }
                        }
                    />
                    </Group>
                    <Group position='apart'>
                    <Text fz="xs">Embedding Model:</Text>
                    <NativeSelect
                        data={Array.from(availableEmbeddings)}
                        defaultValue={formValues.embeddingModel}
                        onChange={(event) => {
                            setFormValues(prev => ({
                                ...prev,    
                                embeddingModel: event.currentTarget.value, 
                            }));
                        }
                        }
                    />
                    </Group>
                    <Group position='right'>
                        <Button type='submit'>Submit</Button>
                    </Group>
                </Stack > 
            </form>
         </Modal>
    );
}