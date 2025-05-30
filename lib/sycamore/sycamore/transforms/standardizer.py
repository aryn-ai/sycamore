from abc import ABC, abstractmethod
from datetime import datetime
import re
from typing import Any, List, Optional

from sycamore.plan_nodes import Node
from sycamore.data import Document
from sycamore.transforms.map import Map

import logging

logger = logging.getLogger(__name__)


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

    # Regexes for military time stuff below.  Example matching strings:
    # clock: 8:00 12:30 23:59:59
    # year: 1970-04-30 1999-12 12/5/2024 12/2000 4/30/70
    # digitpair: 0800 235959
    clock_re = re.compile(r"\d:[0-5]\d")
    year_re = re.compile(r"([12]\d\d\d-)|(/[12]\d\d\d)|(\d/[0-3]?\d/\d)")
    digitpair_re = re.compile(r"([0-2]\d)([0-5]\d)(\d\d)?")

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
        import dateparser

        assert raw_dateTime is not None, "raw_dateTime is None"
        try:
            raw_dateTime = DateTimeStandardizer.fix_military(raw_dateTime)
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
    def fix_military(raw: str) -> str:
        # Fix up military clock time with just digits (0800)
        raw = raw.strip()
        tokens = raw.split()
        saw_clock = 0
        saw_year = 0
        saw_digits = 0
        for token in tokens:
            if DateTimeStandardizer.clock_re.search(token):
                saw_clock += 1
            elif DateTimeStandardizer.year_re.search(token):
                saw_year += 1
            elif DateTimeStandardizer.digitpair_re.fullmatch(token):
                saw_digits += 1
        # If unsure there's exactly one military clock time, bail out.
        # Note that numbers like 2024 could be times or years.
        if (saw_clock > 0) or (saw_year == 0) or (saw_digits != 1):
            return raw
        pieces: list[str] = []
        for token in tokens:
            if match := DateTimeStandardizer.digitpair_re.fullmatch(token):
                clock = ":".join([x for x in match.groups() if x])
                before = token[: match.start(0)]
                after = token[match.end(0) :]
                token = before + clock + after
            pieces.append(token)
        return " ".join(pieces)

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


def ignore_errors(doc: Document, standardizer: Standardizer, key_path: list[str]) -> Document:
    """
    A class for applying the behavior of a standardizer to log errors and continue when encountering null values.

    This class allows for the execution of standardization logic not to fail when encountering null key:value pairs.
    It will instead log a warning stating what key:value pairs in what documents were missing.

    Example:
        .. code-block:: python

            docset.map(lambda doc: ignore_errors(doc, DateTimeStandardizer, ["properties", "entity", "dateAndTime"])
    """

    try:
        doc = standardizer.standardize(doc, key_path=key_path)
    except KeyError:
        logger.warning(f"Key {key_path} not found in document: {doc}")
    except Exception as e:
        logger.error(e)
    return doc
