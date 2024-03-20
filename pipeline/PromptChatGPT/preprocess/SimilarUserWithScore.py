import sys


class SimilarUserWithScore:

    def __init__(self, user_i_id, user_j_id, euclidean_distance):
        self.user_i_id = user_i_id
        self.user_j_id = user_j_id
        self.euclidean_distance = euclidean_distance
        self.rank = sys.maxsize
