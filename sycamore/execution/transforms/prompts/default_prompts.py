ENTITY_EXTRACTOR_GUIDANCE_PROMPT_CHAT = """
    {{#system~}}
    You are a helpful entity extractor.
    {{~/system}}

    {{#user~}}
    You are given a few text elements. The {{entity}} of the file is in these few text elements.Using this context,
    FIND,COPY, and RETURN the {{entity}}. DO NOT REPHRASE OR MAKE UP AN ANSWER.
    {{examples}}
    {{query}}
    {{~/user}}

    {{#assistant~}}
    {{gen "answer"}}
    {{~/assistant}}
    """

ENTITY_EXTRACTOR_GUIDANCE_PROMPT = """
    You are given a few text elements. The {{entity}} of the file is in these few text elements.Using this context,
    FIND,COPY, and RETURN the {{entity}}. DO NOT REPHRASE OR MAKE UP AN ANSWER.
    {{examples}}
    {{query}}
    =========
    {{gen "answer"}}
    """

TEXT_SUMMARIZER_GUIDANCE_PROMPT_CHAT = """
    {{#system~}}
    You are a helpful text summarizer.
    {{~/system}}

    {{#user~}}
    Write a summary of the following. Use only the information provided.
    Include as many key details as possible. Do not make up answer."
    {{query}}
    {{~/user}}

    {{#assistant~}}
    {{gen "summary"}}
    {{~/assistant}}
    """
TEXT_SUMMARIZER_GUIDANCE_PROMPT = """
    Write a summary of the following. Use only the information provided.
    Include as many key details as possible. Do not make up answer."
    {{query}}
    =========
    {{gen "summary"}}
    """
