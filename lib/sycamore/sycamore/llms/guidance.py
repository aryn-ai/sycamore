from sycamore.llms.prompts.default_prompts import SimplePrompt
from guidance.models import Chat, Instruct, Model as GuidanceModel
from guidance import gen, user, system, assistant, instruction


def _execute_chat(prompt: SimplePrompt, model: GuidanceModel, **kwargs) -> str:
    with system():
        lm = model + prompt.system.format(**kwargs)

    with user():
        lm += prompt.user.format(**kwargs)

    with assistant():
        lm += gen(name=prompt.var_name)

    return lm[prompt.var_name]


def _execute_instruct(prompt: SimplePrompt, model: GuidanceModel, **kwargs) -> str:
    with instruction():
        lm = model + prompt.user.format(**kwargs)
    lm += gen(name=prompt.var_name)
    return lm[prompt.var_name]


def _execute_completion(prompt: SimplePrompt, model: GuidanceModel, **kwargs) -> str:
    lm = model + prompt.user.format(**kwargs) + gen(name=prompt.var_name)
    return lm[prompt.var_name]


def execute_with_guidance(prompt: SimplePrompt, model: GuidanceModel, **kwargs) -> str:
    if isinstance(model, Chat):
        return _execute_chat(prompt, model, **kwargs)
    elif isinstance(model, Instruct):
        return _execute_instruct(prompt, model, **kwargs)
    else:
        return _execute_completion(prompt, model, **kwargs)
