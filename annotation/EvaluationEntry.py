class EvaluationEntry:

    def __init__(self, user_id, question, approachName, ground_truth_stance, ground_truth_answer):

        self.user_id = user_id
        self.question = question
        self.approachName = approachName
        self.ground_truth_stance = ground_truth_stance
        self.ground_truth_answer = ground_truth_answer

        self.predicted_stance = None
        self.predicted_answer = None
