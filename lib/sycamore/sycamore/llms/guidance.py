from typing import TYPE_CHECKING

from sycamore.llms.prompts.default_prompts import SimplePrompt

if TYPE_CHECKING:
    from guidance.models import Model as GuidanceModel


def _execute_chat(prompt: SimplePrompt, model: "GuidanceModel", **kwargs) -> str:
    from guidance import gen, user, system, assistant

    with system():
        assert prompt.system is not None
        lm = model + prompt.system.format(**kwargs)

    with user():
        assert prompt.user is not None
        lm += prompt.user.format(**kwargs)

    with assistant():
        lm += gen(name=prompt.var_name)

    return lm[prompt.var_name]


def _execute_instruct(prompt: SimplePrompt, model: "GuidanceModel", **kwargs) -> str:
    from guidance import gen, instruction

    with instruction():
        assert prompt.user is not None
        lm = model + prompt.user.format(**kwargs)
    lm += gen(name=prompt.var_name)
    return lm[prompt.var_name]


def _execute_completion(prompt: SimplePrompt, model: "GuidanceModel", **kwargs) -> str:
    from guidance import gen

    assert prompt.user is not None
    lm = model + prompt.user.format(**kwargs) + gen(name=prompt.var_name)
    return lm[prompt.var_name]


def execute_with_guidance(prompt: SimplePrompt, model: "GuidanceModel", **kwargs) -> str:
    from guidance.models import Chat, Instruct

    if isinstance(model, Chat):
        return _execute_chat(prompt, model, **kwargs)
    elif isinstance(model, Instruct):
        return _execute_instruct(prompt, model, **kwargs)
    else:
        return _execute_completion(prompt, model, **kwargs)
