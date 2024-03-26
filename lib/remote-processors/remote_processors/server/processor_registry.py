from typing import Union
from remote_processors.processors import RequestProcessor, ResponseProcessor


class ProcessorRegistry:
    """Class to hold references to all the processor classes by name for
    easy lookup and use during pipeline configuration
    """

    def __init__(self):
        """Collects the processors by looking at the subclasses of RequestProcessor and ResponseProcessor.
        Note that this only loads processor classes imported (or in files imported) by lib/processors/__init__.py

        Raises:
            DuplicatedProcessorNameError: two processors cannot have the same name
        """
        all_subclasses = [c for c in RequestProcessor.__subclasses__() + ResponseProcessor.__subclasses__()]
        names = {c.get_class_name() for c in all_subclasses}
        if len(names) < len(all_subclasses):
            raise DuplicatedProcessorNameError(all_subclasses)

        self._request_processors: dict[str, type[RequestProcessor]] = {
            c.get_class_name(): c for c in RequestProcessor.__subclasses__()
        }
        self._response_processors: dict[str, type[ResponseProcessor]] = {
            c.get_class_name(): c for c in ResponseProcessor.__subclasses__()
        }

    def get_processor(self, name: str) -> Union[type[RequestProcessor], type[ResponseProcessor], None]:
        if name in self._request_processors:
            return self._request_processors[name]
        if name in self._response_processors:
            return self._response_processors[name]
        return None


class DuplicatedProcessorNameError(Exception):
    """Two processors may not have the same name"""

    def __init__(self, classes: list[Union[type[RequestProcessor], type[ResponseProcessor]]], *args) -> None:
        """Builds message that shows all duplicated processor names

        Args:
            classes (list[Union[RequestProcessor, ResponseProcessor]]): List of all found processor classes
        """
        super().__init__(*args)
        seen = set()
        duplicate_names = []
        for c in classes:
            if c.get_class_name() not in seen:
                seen.add(c.get_class_name())
            else:
                duplicate_names.append(c.get_class_name())

        self._msg = "Duplicated processor names: "
        for name in duplicate_names:
            dupe_classes = [c for c in classes if c.get_class_name() == name]
            message_segment = name + ":\n\t"
            message_segment += "\n\t".join([dc.__name__ for dc in dupe_classes])
            self._msg += "\n" + message_segment

    def __str__(self):
        return self._msg
