from typing import Literal, Union

import torch


class ExternOpinionArgument:
    def __init__(self, argument_id, link, question_text, category, tags, user, stance, argument_title, text, conclusion):
        self.argument_id: str = argument_id
        self.link: str = link
        self.question_text: str = question_text
        self.category: str = category
        self.tags: str = tags
        self.user: str = user
        self.stance: str = stance
        self.argument_title: str = argument_title
        self.text: str = text
        self.conclusion: str = conclusion

        self.vector_representation = dict()

    def set_vector_representation(
            self, for_target: Literal["argument_title", "conclusion", "premise", "whole_argument"],
            vector_representation
    ):
        self.vector_representation[for_target] = vector_representation

    def to_text(
            self, target: Literal["argument_title", "conclusion", "premise", "whole_argument"]
    ) -> Union[str, torch.Tensor]:
        if target in self.vector_representation:
            return self.vector_representation[target]

        if target == "argument_title":
            return self.argument_title
        elif target == "conclusion":
            return self.conclusion
        elif target == "premise":
            return self.text
        elif target == "whole_argument":
            ret = self.text
            if self.conclusion.startswith("Therefore"):
                ret += " " + self.conclusion
            else:
                ret += " Therefore, " + self.conclusion
            return ret

        raise ValueError("Unknown target: {}".format(target))

    def __str__(self):
        return "{}-Argument {} by \"{}\": {}\n({}) {}".format(
            self.stance.upper(), self.argument_id, self.user, self.question_text, self.argument_title, self.text
        )

    def __hash__(self):
        return hash(self.argument_id)

    def __eq__(self, other):
        return isinstance(other, ExternOpinionArgument) and self.argument_id == other.argument_id
