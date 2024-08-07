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
    An abstract class for implementing standardizers.
    """

    @abstractmethod
    def fixer(self, text: str) -> Union[str, Tuple[str, date]]:
        pass

    def standardize(self, doc: Document, key_path: List[str]) -> Document:
        current = doc
        for key in key_path[:-1]:
            if key in current.keys():
                current = current[key]
            else:
                raise KeyError(f"Key {key} not found in the dictionary among {current.keys()}")
        target_key = key_path[-1]
        if target_key in current.keys():
            current[target_key] = self.fixer(current[target_key])
        else:
            raise KeyError(f"Key {target_key} not found in the dictionary among {current.keys()}")
        return doc


class LocationStandardizer(Standardizer):
    """
    This Class standardizes the format of US State abbreviations.
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
        This method replaces the US State abbreviations with full names.
        """

        def replacer(match):
            abbreviation = match.group(0)
            return LocationStandardizer.state_abbreviations.get(abbreviation, abbreviation)

        return re.sub(r"\b[A-Z]{2}\b", replacer, text)


class DateTimeStandardizer(Standardizer):
    """
    This Class standardizes the format of dateTime.
    """

    def fixer(self, raw_dateTime: str) -> Tuple[str, date]:
        """
        This method standardizes the datetime property of elements by replacing
        periods with colons and parsing the date as a Date object.
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
    A Class for implementing Standardizers. This class runs the
    standardizer for location or datetime on a DocSet.

    """

    def __init__(
        self,
        child: Node,
        standardizer: Standardizer,
        path: list[str],
    ):
        super().__init__(child, f=standardizer.standardize, args=path)
