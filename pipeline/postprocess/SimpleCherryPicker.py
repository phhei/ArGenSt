from typing import Any, List
from random import randint

from pipeline.postprocess.CherryPicker_Base import CherryPickerInterface


class FirstPicker(CherryPickerInterface):
    def cherry_pick(self, user: Any, topic: str, stance_probability: float, candidates: List[str]) -> int:
        return 0

    def __str__(self) -> str:
        return "Pick First"


class RandomPicker(CherryPickerInterface):
    def cherry_pick(self, user: Any, topic: str, stance_probability: float, candidates: List[str]) -> int:
        return randint(a=0, b=len(candidates)-1)

    def __str__(self) -> str:
        return "Pick random"


class LengthCheryPicker(CherryPickerInterface):
    def cherry_pick(self, user: Any, topic: str, stance_probability: float, candidates: List[str]) -> int:
        length_list = [(i, len(candidate)) for i, candidate in enumerate(candidates)]
        length_list.sort(key=lambda c: c[1], reverse=True)
        return length_list[0][0]

    def __str__(self) -> str:
        return "Pick longest"
