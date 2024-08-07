from abc import ABC, abstractmethod

from sycamore.plan_nodes import Node

from sycamore.data import Document
from sycamore.transforms.map import Map
from dateutil import parser
import re
from typing import List


class Standardizer(ABC):
    """
    An Abstrant class for implementing Standarizers.
    """

    @abstractmethod
    def standardize(self, doc: Document, key_path: List[str]) -> Document:
        pass


class LocationStandardizer(Standardizer):
    """
    This Class standardizes the format of US State abbreviations.
    """

    def __init__(self):
        self._state_abbreviations = {
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

    def replace_abbreviations(self, input_string):
        """
        This method replaces the US State abbreviations with full names.
        """
        state_abbreviations = {key: value for key, value in self._state_abbreviations.items()}
        abbreviations = {**state_abbreviations}

        pattern = re.compile(r"\b(" + "|".join(re.escape(key) for key in abbreviations.keys()) + r")\b")

        def replace_match(match):
            return abbreviations[match.group(0)]

        result_string = pattern.sub(replace_match, input_string)
        return result_string

    def standardize(self, doc: Document, key_path: List[str]) -> Document:
        """
        This Methods creates a new element standardises the location property of the Element
        by replcaing all instance of US State abbreviation with correct name.
        """
        current = doc
        for key in key_path[:-1]:
            if key in current.keys():
                current = current[key]
            else:
                raise KeyError(f"Key {key} not found in the dictionary among {current.keys()}")
        target_key = key_path[-1]
        if target_key in current.keys():
            current[target_key] = self.replace_abbreviations(current[target_key])
        else:
            raise KeyError(f"Key {target_key} not found in the dictionary among {current.keys()}")
        return doc


class DateTimeStandardizer(Standardizer):
    """
    This Class standardizes the format of dateTime.
    """

    def fix_date(self, raw_dateTime):
        """
        This Method standardizes the datetime property of Elements by replcaing . with : and parsing date as Date
        """

        raw_dateTime = raw_dateTime.replace("Local", "")
        raw_dateTime = raw_dateTime.replace(".", ":")
        parsed_date = parser.parse(raw_dateTime)
        extracted_date = parsed_date.date()
        return raw_dateTime, extracted_date

    def standardize(self, doc: Document, key_path) -> Document:
        """
        This Methods creates a new element with Date and standardises the dateTime property of the Element
        """

        current = doc
        for key in key_path[:-1]:
            if key in current.keys():
                current = current[key]
            else:
                raise KeyError(f"Key {key} not found in the dictionary among {current.keys()}")
        target_key = key_path[-1]
        if target_key in current.keys():
            current[target_key], current["day"] = self.fix_date(current[target_key])
        else:
            raise KeyError(f"Key {target_key} not found in the dictionary among {current.keys()}")
        return doc


class StandardizeProperty(Map):
    """
    The Class runs the standarizer iether for location or datetime on DocSet.
    """

    def __init__(
        self,
        child: Node,
        standardizer: Standardizer,
        path: list[str],
    ):
        super().__init__(child, f=standardizer.standardize, args=path)
