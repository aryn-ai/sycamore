from typing import Union
from lib.processors import RequestProcessor, ResponseProcessor



class ProcessorRegistry:

    def __init__(self):
        all_subclasses = [c for c in RequestProcessor.__subclasses__() + ResponseProcessor.__subclasses__()]
        names = {c.get_class_name() for c in all_subclasses}
        if len(names) < len(all_subclasses):
            raise DuplicatedProcessorNameError(all_subclasses)

        self._request_processors: dict[str, type[RequestProcessor]] = \
            {c.get_class_name(): c for c in RequestProcessor.__subclasses__()}
        self._response_processors: dict[str, type[ResponseProcessor]] = \
            {c.get_class_name(): c for c in ResponseProcessor.__subclasses__()}
        
    def get_processor(self, name: str) -> Union[RequestProcessor, ResponseProcessor, None]:
        if name in self._request_processors:
            return self._request_processors[name]
        if name in self._response_processors:
            return self._response_processors[name]
        return None
    

    
class DuplicatedProcessorNameError(Exception):

    def __init__(self, classes: list[Union[RequestProcessor, ResponseProcessor]], *args) -> None:
        super().__init__(*args)
        seen = set()
        duplicate_names = [c.get_class_name() for c in classes \
                           if c.get_class_name() in seen or seen.add(c.get_class_name())]
        self._msg = "Duplicated processor names: "
        for name in duplicate_names:
            dupe_classes = [c for c in classes if c.get_class_name() == name]
            message_segment = name + ":\n\t"
            message_segment += "\n\t".join([dc.__name__ for dc in dupe_classes])
            self._msg += "\n" + message_segment

    def __str__(self):
        return self._msg