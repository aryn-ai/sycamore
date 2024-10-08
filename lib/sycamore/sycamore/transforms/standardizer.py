from abc import ABC, abstractmethod
from datetime import datetime
import re
from typing import Any, List, Optional

import dateparser
from sycamore.plan_nodes import Node
from sycamore.data import Document
from sycamore.transforms.map import Map


class Standardizer(ABC):
    """
    An abstract base class for implementing standardizers, which are responsible for
    transforming specific fields within a document according to certain rules.

    """

    @abstractmethod
    def fixer(self, text: str) -> Any:
        """
        Abstract method to be implemented by subclasses to define how the relevant values
        should be standardized.

        Args:
            text (str): The text or date string to be standardized.

        Returns:
            A standardized value.
        """
        pass

    @abstractmethod
    def standardize(self, doc: Document, key_path: List[str]) -> Document:
        """
        Abstract method applies the fixer method to a specific field in the document as defined by the key_path.

        Args:
            doc (Document): The document to be standardized.
            key_path (List[str]): The path to the field within the document that should be standardized.

        Returns:
            Document: The document with the standardized field.

        Raises:
            KeyError: If any of the keys in key_path are not found in the document.
        """
        pass


class USStateStandardizer(Standardizer):
    """
    A standardizer for transforming US state abbreviations in text to their full state names.
    Transforms substrings matching a state abbreviation to the full state name.

    Example:
        .. code-block:: python

            source_docset = ...  # Define a source node or component that provides hierarchical documents.
            transformed_docset = source_docset.map(
                lambda doc: USStateStandardizer.standardize(
                    doc,
                    key_path = ["path","to","location"]))

    """

    state_abbreviations = {
        "AK": "Alaska",
        "AL": "Alabama",
        "AR": "Arkansas",
        "AZ": "Arizona",
        "CA": "California",
        "CO": "Colorado",
        "CT": "Connecticut",
        "DC": "District of Columbia",
        "DE": "Delaware",
        "FL": "Florida",
        "GA": "Georgia",
        "HI": "Hawaii",
        "IA": "Iowa",
        "ID": "Idaho",
        "IL": "Illinois",
        "IN": "Indiana",
        "KS": "Kansas",
        "KY": "Kentucky",
        "LA": "Louisiana",
        "MA": "Massachusetts",
        "MD": "Maryland",
        "ME": "Maine",
        "MI": "Michigan",
        "MN": "Minnesota",
        "MO": "Missouri",
        "MS": "Mississippi",
        "MT": "Montana",
        "NC": "North Carolina",
        "ND": "North Dakota",
        "NE": "Nebraska",
        "NH": "New Hampshire",
        "NJ": "New Jersey",
        "NM": "New Mexico",
        "NV": "Nevada",
        "NY": "New York",
        "OH": "Ohio",
        "OK": "Oklahoma",
        "OR": "Oregon",
        "PA": "Pennsylvania",
        "RI": "Rhode Island",
        "SC": "South Carolina",
        "SD": "South Dakota",
        "TN": "Tennessee",
        "TX": "Texas",
        "UT": "Utah",
        "VA": "Virginia",
        "VT": "Vermont",
        "WA": "Washington",
        "WI": "Wisconsin",
        "WV": "West Virginia",
        "WY": "Wyoming",
    }

    @staticmethod
    def fixer(text: str) -> str:
        """
        Replaces any US state abbreviations in the text with their full state names.

        Args:
            text (str): The text containing US state abbreviations.

        Returns:
            str: The text with state abbreviations replaced by full state names.
        """

        def replacer(match):
            abbreviation = match.group(0)
            return USStateStandardizer.state_abbreviations.get(abbreviation, abbreviation)

        return re.sub(r"\b[A-Z]{2}\b", replacer, text)

    @staticmethod
    def standardize(doc: Document, key_path: List[str]) -> Document:
        """
        Applies the fixer method to a specific field in the document as defined by the key_path.

        Args:
            doc (Document): The document to be standardized.
            key_path (List[str]): The path to the field within the document that should be standardized.

        Returns:
            Document: The document with the standardized field.

        Raises:
            KeyError: If any of the keys in key_path are not found in the document.
        """
        current = doc
        for key in key_path[:-1]:
            if current.get(key, None):
                current = current[key]
            else:
                raise KeyError(f"Key {key} not found in the dictionary among {current.keys()}")
        target_key = key_path[-1]
        if current.get(target_key, None):
            current[target_key] = USStateStandardizer.fixer(current[target_key])
        else:
            raise KeyError(f"Key {target_key} not found in the dictionary among {current.keys()}")
        return doc


class DateTimeStandardizer(Standardizer):
    """
    A standardizer for transforming date and time strings into a consistent format.


    Example:
        .. code-block:: python

            source_docset = ...  # Define a source node or component that provides hierarchical documents.
            transformed_docset = source_docset.map(
                lambda doc: USStateStandardizer.standardize(
                    doc,
                    key_path = ["path","to","datetime"]))
    """

    DEFAULT_FORMAT = "%B %d, %Y %H:%M:%S%Z"

    @staticmethod
    def fixer(raw_dateTime: str) -> datetime:
        """
        Standardize a date-time string by parsing it into a datetime object.

        Args:
            raw_dateTime (str): The raw date-time string to be standardized.
            format: Optional[str]: strftime-compatible format string to render the datetime.

        Returns:
            Tuple[str, date]: A tuple containing the standardized date-time string and the corresponding
            datetime object.

        Raises:
            ValueError: If the input string cannot be parsed into a valid date-time.
            RuntimeError: For any other unexpected errors during the processing.
        """
        assert raw_dateTime is not None, "raw_dateTime is None"
        try:
            raw_dateTime = raw_dateTime.strip()
            raw_dateTime = raw_dateTime.replace("Local", "")
            raw_dateTime = raw_dateTime.replace("local", "")
            raw_dateTime = raw_dateTime.replace(".", ":")
            parsed = dateparser.parse(raw_dateTime)
            if not parsed:
                raise ValueError(f"Invalid date format: {raw_dateTime}")
            return parsed

        except ValueError as e:
            # Handle errors related to value parsing
            raise ValueError(f"Invalid date format: {raw_dateTime}") from e

        except Exception as e:
            # Handle any other exceptions
            raise RuntimeError(f"Unexpected error occurred while processing: {raw_dateTime}") from e

    @staticmethod
    def standardize(
        doc: Document,
        key_path: List[str],
        add_day: bool = True,
        add_dateTime: bool = True,
        date_format: Optional[str] = None,
    ) -> Document:
        """
        Applies the fixer method to a specific date-time field in the document as defined by the key_path.

        Args:
            doc (Document): The document to be standardized.
            key_path (List[str]): The path to the date-time field within the document that should be
                standardized.
            add_day (bool): Whether to add a "day" field to the document with the date extracted from the
                standardized date-time field. Will not overwrite an existing "day" field.
            add_dateTime (bool): Whether to add a "dateTime" field to the document with the standardized
                standardized date-time field. Will not overwrite an existing "dateTime" field.
            date_format (Optional[str]): strftime-compatible format string to render the datetime.

        Returns:
            Document: The document with the standardized date-time field and an additional "day" field.

        Raises:
            KeyError: If any of the keys in key_path are not found in the document.
        """
        current = doc
        for key in key_path[:-1]:
            if key in current.keys():
                current = current[key]
            else:
                raise KeyError(f"Key {key} not found in the dictionary among {current.keys()}")
        target_key = key_path[-1]
        if target_key not in current.keys():
            raise KeyError(f"Key {target_key} not found in the dictionary among {current.keys()}")
        if current[target_key] is None:
            raise KeyError(f"Key {target_key} has value None")

        parsed = DateTimeStandardizer.fixer(current[target_key])
        rendered = parsed.strftime(date_format or DateTimeStandardizer.DEFAULT_FORMAT)
        current[target_key] = rendered
        if add_dateTime and "dateTime" not in current.keys():
            current["dateTime"] = parsed
        if add_day and "day" not in current.keys():
            current["day"] = parsed.date()
        return doc


class StandardizeProperty(Map):
    """
    A class for applying a standardizer to a specific property of documents in a dataset.

    This class allows for the execution of standardization logic, either for location or date-time
    properties, across a set of documents by utilizing a specified standardizer and path.
    """

    def __init__(
        self,
        child: Node,
        standardizer: Standardizer,
        path: list[str],
        **kwargs,
    ):
        super().__init__(child, f=standardizer.standardize, args=path, kwargs=kwargs)
