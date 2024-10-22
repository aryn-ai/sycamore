import sys
from typing import Optional
from sycamore.utils.pickle_pydantic import safe_cloudpickle, safe_cloudunpickle
from pydantic import BaseModel, Field


def test_pydantic_picklng() -> None:
    class BoardMember(BaseModel):
        name: str
        votes_for: Optional[int]
        votes_against: Optional[int]
        votes_abstentions: Optional[int]

    class Company(BaseModel):
        name: str
        founded: Optional[str] = Field(description="The date a company was founded")

    company_pickled = safe_cloudpickle(Company)
    board_member_pickled = safe_cloudpickle(BoardMember)

    company_unpickled = safe_cloudunpickle(company_pickled)
    board_member_unpickled = safe_cloudunpickle(board_member_pickled)

    assert sys.getsizeof(company_unpickled) == sys.getsizeof(Company)
    assert sys.getsizeof(board_member_unpickled) == sys.getsizeof(BoardMember)
