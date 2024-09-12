from typing import Optional

import sycamore
from sycamore.context import context_params, Context, get_val_from_context, ExecMode


def test_init():
    from sycamore.rules.optimize_resource_args import EnforceResourceUsage, OptimizeResourceArgs

    context = sycamore.init()

    assert context is not None
    assert len(context.rewrite_rules) == 2
    assert isinstance(context.rewrite_rules[0], EnforceResourceUsage)
    assert isinstance(context.rewrite_rules[1], OptimizeResourceArgs)

    another_context = sycamore.init()
    assert another_context is not context


def test_get_val_from_context():
    params = {"paramKeyA": {"llm": 1}, "paramKeyB": {"llm": ["llm1", "llm2"]}, "default": {"llm": "openai"}}
    context = Context(params=params)

    assert "openai" == get_val_from_context(context, "llm")
    assert 1 == get_val_from_context(context, "llm", param_names=["paramKeyA"])
    assert ["llm1", "llm2"] == get_val_from_context(context, "llm", param_names=["paramKeyB"])
    assert 1 == get_val_from_context(
        context, "llm", param_names=["paramKeyA", "paramKeyB"]
    ), "Unable to assert ordered parameter name retrieval"
    assert get_val_from_context(context, "missing") is None
    assert get_val_from_context(Context(), "missing") is None
    assert get_val_from_context(Context(params={}), "missing") is None


@context_params
def get_first_character(some_function_arg: str, context: Optional[Context] = None):
    assert some_function_arg is not None
    return some_function_arg[0]


class ClassThatContainsAContext:

    def __init__(self, context: Context) -> None:
        super().__init__()
        self.context = context

    @context_params
    def get_first_character_w_class_context(self, some_function_arg: str):
        """
        This method should verify that we're able to use the class's context var if the method doesn't accept one
        """
        assert some_function_arg is not None
        return some_function_arg[0]


def test_function_w_context():
    context = Context(params={"default": {"some_function_arg": "Aryn"}})
    assert "A" == get_first_character(context=context)

    # ensure explicit arg overrides context
    assert "B" == get_first_character("Baryn", context=context)
    assert "C" == get_first_character("Caryn")


def test_function_w_class_context():
    context = Context(params={"default": {"some_function_arg": "Aryn"}})
    obj = ClassThatContainsAContext(context)

    assert "A" == obj.get_first_character_w_class_context()

    # ensure explicit arg overrides context
    assert "B" == obj.get_first_character_w_class_context("Baryn")


@context_params
def two_positional_args_method(some_function_arg: str, some_other_arg: str, context: Optional[Context] = None):
    assert some_function_arg is not None
    assert some_other_arg is not None
    return some_function_arg + " " + some_other_arg


@context_params
def two_positional_args_method_with_kwargs(some_function_arg: str, context: Optional[Context] = None, **kwargs):
    assert some_function_arg is not None
    assert kwargs.get("some_other_arg") is not None
    return some_function_arg + " " + str(kwargs.get("some_other_arg"))


def test_positional_args_and_context_args():
    context = Context(
        params={"default": {"some_other_arg": "Aryn2", "some_unrelated_arg": "ArynZ"}}, exec_mode=ExecMode.LOCAL
    )

    # no context
    assert "a b" == two_positional_args_method("a", "b")

    # Should ignore context vars because of positional args
    assert "a b" == two_positional_args_method("a", "b", context=context)

    # Pickup 'some_other_arg' from context
    assert "a Aryn2" == two_positional_args_method(some_function_arg="a", context=context)

    # Should ignore context vars because of kwargs
    assert "a b" == two_positional_args_method(some_function_arg="a", some_other_arg="b", context=context)


    # Combine positional and kwarg
    assert "a b" == two_positional_args_method_with_kwargs("a", some_other_arg="b", context=context)


def test_positional_args_and_context_args_f_with_kwargs():
    context = Context(
        params={"default": {"some_other_arg": "Aryn2", "some_unrelated_arg": "ArynZ"}}, exec_mode=ExecMode.LOCAL
    )
    # Pickup 'some_other_arg' from context
    assert "a Aryn2" == two_positional_args_method_with_kwargs(some_function_arg="a", context=context)

    # Should ignore context vars because of kwargs
    assert "a b" == two_positional_args_method_with_kwargs(some_function_arg="a", some_other_arg="b", context=context)

    # Combine positional and kwarg
    assert "a b" == two_positional_args_method_with_kwargs("a", some_other_arg="b", context=context)
