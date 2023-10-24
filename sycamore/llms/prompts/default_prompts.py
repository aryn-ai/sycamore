ENTITY_EXTRACTOR_ZERO_SHOT_GUIDANCE_PROMPT_CHAT = """
    {{#system~}}
    You are a helpful entity extractor.
    {{~/system}}

    {{#user~}}
    You are given a few text elements of a document. The {{entity}} of the document is in these few text elements.Using
    this context,
    FIND,COPY, and RETURN the {{entity}}. DO NOT REPHRASE OR MAKE UP AN ANSWER.
    {{query}}
    {{~/user}}

    {{#assistant~}}
    {{gen "answer"}}
    {{~/assistant}}
    """

ENTITY_EXTRACTOR_ZERO_SHOT_GUIDANCE_PROMPT = """
    You are given a few text elements of a document. The {{entity}} of the document is in these few text elements.Using
    this context,
    FIND,COPY, and RETURN the {{entity}}. DO NOT REPHRASE OR MAKE UP AN ANSWER.
    {{query}}
    =========
    {{gen "answer"}}
    """


ENTITY_EXTRACTOR_FEW_SHOT_GUIDANCE_PROMPT_CHAT = """
    {{#system~}}
    You are a helpful entity extractor.
    {{~/system}}

    {{#user~}}
    You are given a few text elements of a document. The {{entity}} of the document is in these few text elements.Using
    this context, FIND,COPY, and RETURN the {{entity}}. Only return the {{entity}} as part of your answer. DO NOT
    REPHRASE OR MAKE UP AN ANSWER.
    {{query}}
    {{~/user}}

    {{#assistant~}}
    {{gen "answer"}}
    {{~/assistant}}
    """

ENTITY_EXTRACTOR_FEW_SHOT_GUIDANCE_PROMPT = """
    You are given a few text elements of a document. The {{entity}} of the document is in these few text elements.Using
    this context,
    FIND,COPY, and RETURN the {{entity}}. DO NOT REPHRASE OR MAKE UP AN ANSWER.
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
    Include as many key details as possible. Do not make up answer. Only return the summary as part of your answer."
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

SCHEMA_ZERO_SHOT_GUIDANCE_PROMPT_CHAT = """
    {{#system~}}
    You are a helpful entity extractor.
    {{~/system}}

    {{#user~}}
    You are given a few text elements of a document. Extract JSON representing an entity of class {{entity}} from the document.
    Using this context, FIND, FORMAT, and RETURN the {{entity}} JSON. Only return JSON as part of your answer.
    {{query}}
    {{~/user}}

    {{#assistant~}}
    {{gen "answer"}}
    {{~/assistant}}
    """

SCHEMA_ZERO_SHOT_GUIDANCE_PROMPT = """
    You are given a few text elements of a document. Extract JSON representing an entity of class {{entity}} from the document.
    Using this context, FIND, FORMAT, and RETURN the {{entity}} JSON. Only return JSON as part of your answer.
    {{query}}
    =========
    {{gen "answer"}}
    """
