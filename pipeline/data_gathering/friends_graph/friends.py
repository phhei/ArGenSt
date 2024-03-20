class Friends:

    def __init__(self, user_id_1, user_id_2, friendship_strength):
        self.user_id_1 = user_id_1
        self.user_id_2 = user_id_2
        self.friendship_strength = friendship_strength

    def __hash__(self) -> int:
        return hash(self.user_id_1) + hash(self.user_id_2) - hash(self.friendship_strength)

    def __eq__(self, other) -> bool:
        return isinstance(other, Friends) and self.friendship_strength == other.friendship_strength and \
               ((self.user_id_1 == other.user_id_1 and self.user_id_2 == other.user_id_2) or
                (self.user_id_1 == other.user_id_2 and self.user_id_2 == other.user_id_1))
