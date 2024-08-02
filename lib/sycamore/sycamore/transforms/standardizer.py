from typing import Dict, List, Union, Any, Tuple
from abc import ABC, abstractmethod

from sycamore.plan_nodes import Node

import os
from sycamore.data import Element, Document
from sycamore import Context
from sycamore.transforms.map import Map
from dateutil import parser


class Standardizer(ABC):

    @abstractmethod
    def standardize(self ,document: Document) -> Document:
        pass


class LocationStandardizer(Standardizer):
    """
    This Class standardizes the format of location.
    Attributes:
        
    """

    def __init__(
        self
    ):
        self._state_dict = {
    "AK": "Alaska", "AL": "Alabama", "AR": "Arkansas", "AZ": "Arizona", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DC": "District of Columbia", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "IA": "Iowa", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "MA": "Massachusetts", "MD": "Maryland", "ME": "Maine", "MI": "Michigan", "MN": "Minnesota",
    "MO": "Missouri", "MS": "Mississippi", "MT": "Montana", "NC": "North Carolina", "ND": "North Dakota",
    "NE": "Nebraska", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NV": "Nevada", "NY": "New York", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VA": "Virginia", "VT": "Vermont",
    "WA": "Washington", "WI": "Wisconsin", "WV": "West Virginia", "WY": "Wyoming"}

    def _standardize_state(self, state:str) -> str:
        clean_state = state.lstrip().rstrip().upper()
        return self._state_dict.get(clean_state,"")
    
    def standardize(self, doc:Document) -> Document:
        if "location" not in doc.properties['entity']:
            return doc
        raw_location = doc.properties['entity']['location']
        city, state = raw_location.split(',')
        std_loc = f'{city},  {self._standardize_state(state)}'
        doc.properties['entity']['location'] = std_loc 
        return doc 
        
class DateTimeStandardizer(Standardizer):
    """
    This Class standardizes the format of dateTime.
    Attributes:
        
    """
 
    def standardize(self, doc:Document) -> Document:
        if "dateTime" not in doc.properties['entity']:
            return doc
        raw_dateTime = doc.properties['entity']['dateTime']
        raw_dateTime = raw_dateTime.replace("Local", "")
        raw_dateTime = raw_dateTime.replace(".", ":")
        parsed_date = parser.parse(raw_dateTime)
        extracted_date = parsed_date.date()
        doc.properties['entity']['day'] = extracted_date
        return doc 
        


class Standardize_property(Map):
    """
    The Class runs the standarizer iether location or date on DocSet.
    """

    def __init__(
        self,
        child: Node,
        standardizer: Standardizer,
        **resource_args,
    ):
        super().__init__(child, f=standardizer.standardize, **resource_args)