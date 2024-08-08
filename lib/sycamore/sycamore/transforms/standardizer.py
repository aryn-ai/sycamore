from abc import ABC, abstractmethod

from sycamore.plan_nodes import Node

from sycamore.data import Document
from sycamore.transforms.map import Map
from dateutil import parser
import re
from typing import List, Tuple, Union
from datetime import date


class Standardizer(ABC):
    """
    An abstract base class for implementing standardizers, which are responsible for
    transforming specific fields within a document according to certain rules.
    """

    @abstractmethod
    def fixer(self, text: str) -> Union[str, Tuple[str, date]]:
        """
        Abstract method to be implemented by subclasses to define how the relevant values
        should be standardized.

        Args:
            text (str): The text or date string to be standardized.

        Returns:
            Union[str, Tuple[str, date]]: The standardized text or a tuple containing the standardized date string and date.
        """
        pass

    def standardize(self, doc: Document, key_path: List[str]) -> Document:
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
            if current.get(key,None):
                current = current[key]
            else:
                raise KeyError(f"Key {key} not found in the dictionary among {current.keys()}")
        target_key = key_path[-1]
        if current.get(target_key,None):
            current[target_key] = self.fixer(current[target_key])
        else:
            raise KeyError(f"Key {target_key} not found in the dictionary among {current.keys()}")
        return doc


class LocationStandardizer(Standardizer):
    """
    A standardizer for transforming US state abbreviations in text to their full state names.
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

    def fixer(self, text: str) -> str:
        """
        Replaces any US state abbreviations in the text with their full state names.

        Args:
            text (str): The text containing US state abbreviations.

        Returns:
            str: The text with state abbreviations replaced by full state names.
        """

        def replacer(match):
            abbreviation = match.group(0)
            return LocationStandardizer.state_abbreviations.get(abbreviation, abbreviation)

        return re.sub(r"\b[A-Z]{2}\b", replacer, text)


class DateTimeStandardizer(Standardizer):
    """
    A standardizer for transforming date and time strings into a consistent format.
    """

    def fixer(self, raw_dateTime: str) -> Tuple[str, date]:
        """
        Converts a date-time string by replacing periods with colons and parsing it into a date object.

        Args:
            raw_dateTime (str): The raw date-time string to be standardized.

        Returns:
            Tuple[str, date]: A tuple containing the standardized date-time string and the corresponding date object.

        Raises:
            ValueError: If the input string cannot be parsed into a valid date-time.
            RuntimeError: For any other unexpected errors during the processing.
        """
        try:
            # Clean up the raw_dateTime string
            raw_dateTime = raw_dateTime.replace("Local", "")
            raw_dateTime = raw_dateTime.replace(".", ":")

            # Parse the cleaned dateTime string

            parsed_date = parser.parse(raw_dateTime)
            extracted_date = parsed_date.date()
            return raw_dateTime, extracted_date
        
        except ValueError as e:
            # Handle errors related to value parsing
            raise ValueError(f"Invalid date format: {raw_dateTime}") from e

        except Exception as e:
            # Handle any other exceptions
            raise RuntimeError(f"Unexpected error occurred while processing: {raw_dateTime}") from e
        
    def standardize(self, doc: Document, key_path: List[str]) -> Document:
        """
        Applies the fixer method to a specific date-time field in the document as defined by the key_path,
        and adds an additional "day" field with the extracted date.

        Args:
            doc (Document): The document to be standardized.
            key_path (List[str]): The path to the date-time field within the document that should be standardized.

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
        if target_key in current.keys():
            current[target_key], current["day"] = self.fixer(current[target_key])
        else:
            raise KeyError(f"Key {target_key} not found in the dictionary among {current.keys()}")
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
    ):
        super().__init__(child, f=standardizer.standardize, args=path)
