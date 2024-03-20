class Claim:

    def __init__(self, link, title):
        self.link = link
        self.title = title

    def __hash__(self) -> int:
        return hash(self.link) + hash(self.title)

    def __eq__(self, other) -> bool:
        return isinstance(other, Claim) and self.link == other.link and self.title == other.title
