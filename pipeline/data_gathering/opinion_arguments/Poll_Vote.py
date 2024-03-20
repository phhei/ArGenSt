class PollVote:

    def __init__(self, user, link, title, vote, explanation):
        self.user = user
        self.link = link
        self.title = title
        self.vote = vote
        self.explanation = explanation

    def __hash__(self) -> int:
        return hash(self.user) + hash(self.link)

    def __eq__(self, other) -> bool:
        return isinstance(other, PollVote) and self.user == other.user and self.link == other.link
