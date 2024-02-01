## ExtractEntity
The Extract Entity Transform extracts semantically meaningful information from your documents. The ``OpenAIEntityExtractor`` leverages one of OpenAI's LLMs to perform this extraction with just a few examples. These extracted entities are then incorporated as properties into the document structure. The following code shows how to provide an example template for extracting a title using the gpt-3.5-turbo model.

```python
openai_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)
title_prompt_template = """
    ELEMENT 1: Jupiter's Moons
    ELEMENT 2: Ganymede 2020
    ELEMENT 3: by Audi Lauper and Serena K. Goldberg. 2011
    ELEMENT 4: From Wikipedia, the free encyclopedia
    ELEMENT 5: Ganymede, or Jupiter III, is the largest and most massive natural satellite of Jupiter as well as in the Solar System, being a planetary-mass moon. It is the largest Solar System object without an atmosphere, despite being the only moon of the Solar System with a magnetic field. Like Titan, it is larger than the planet Mercury, but has somewhat less surface gravity than Mercury, Io or the Moon.
    =========
    "Ganymede 2020
"""

docset = docset.extract_entity(entity_extractor=OpenAIEntityExtractor("title", llm=openai_llm, prompt_template=title_context_template))
```
