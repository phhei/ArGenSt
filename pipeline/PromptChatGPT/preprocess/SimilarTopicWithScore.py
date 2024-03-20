import sys


class SimilarTopicWithScore:

    def __init__(self, id_original, topic_original, id_target, topic_target, cosine_similarity):
        self.id_original = id_original
        self.topic_original = topic_original
        self.id_target = id_target
        self.topic_target = topic_target
        self.cosine_similarity = cosine_similarity
        self.rank = sys.maxsize
