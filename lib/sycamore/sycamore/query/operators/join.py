from typing import List

from sycamore.query.operators.logical_operator import LogicalOperator


class Join(LogicalOperator):
    """Joins two databases based on a particular field.
        
    Values of *fieldOne* from 
    database 1 are used to filter records of database 2 based on values of *fieldTwo* 
    in database 2. For example, consider that database 1 is {"properties.key": 
    ['Cruise Ship', 'Sailboat'], "properties.count": [3, 2]} and database 2 is 
    {"properties.entity.shipType": ['Jet ski', 'Canoe', 'Submarine', 'Cruise Ship'], 
    "properties.entity.country": ['Australia', 'Japan', 'United States', 'Mexico'], 
    "properties.entity.city": ['Sydney', 'Kyoto', 'San Francisco', 'Cabo']}. A join 
    operation with *inputs* containing ids of operations that return database 1 and 
    database 2, respectively, *fieldOne* being "properties.key", and *fieldTwo* being 
    "properties.entity.shipType", would return the database {"properties.entity.shipType": 
    ['Cruise Ship'], "properties.entity.country": ['Mexico'], "properties.entity.city": 
    ['Cabo']}.
            
    Parameters are *description*, *fieldOne*, *fieldTwo*, *input*, and *id*. 

    Returns a database with fields identical to those in database 2.

    - *description* is a written description of the purpose of this operation in this context 
        and justification of why you chose to use it.
    - *fieldOne* is the the field in the first database.
    - *fieldTwo* is the field in the second database.
    - *input* is a list of operation ids that this operation depends on. For this operation, 
        *input* should only two id of operations that returns a databases (len(input) == 2).
    - *id* is a uniquely assigned integer that serves as an identifier.
    """

    fieldOne: str
    fieldTwo: str
    input: List[str]