from collections import defaultdict
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Dict

import torch.cuda
import transformers

from pipeline.postprocess.CherryPicker_Base import CherryPickerInterface
from loguru import logger


class SentimentCherryPicker(CherryPickerInterface):
    LABEL_MAPPER = {
        "cardiffnlp/twitter-roberta-base-sentiment": {"LABEL_0": -1, "LABEL_1": 0, "LABEL_2": 1},
        "cardiffnlp/twitter-roberta-base-sentiment-latest": {"negative": -1, "neutral": 0, "positive": 1},
        "siebert/sentiment-roberta-large-english": {"NEGATIVE": -1, "POSITIVE": 1},
        "finiteautomata/bertweet-base-sentiment-analysis": {"NEG": -1, "NEU": 0, "POS": 1}
    }
    SCORE_MAPPER = {-1: "NEGATIVE", 0: "UNDECIDED", 1: "POSITIVE"}

    def __init__(
            self,
            root_path: Path,
            models: List[Literal["cardiffnlp/twitter-roberta-base-sentiment",
                                 "siebert/sentiment-roberta-large-english",
                                 "finiteautomata/bertweet-base-sentiment-analysis",
                                 "cardiffnlp/twitter-roberta-base-sentiment-latest"]],
            device: Optional[str] = None,
            ignore_topic: bool = False
    ):
        super().__init__(root_path)

        logger.trace("OK, let's load our {} models", len(models))

        self.text_classifiers: List[Tuple[str, transformers.TextClassificationPipeline]] = []
        for model in ([models] if isinstance(models, str) else models):
            logger.debug("Let's load {}", model)
            self.text_classifiers.append(
                (
                    model,
                    transformers.pipeline(
                        task="text-classification",
                        model=model,
                        framework="pt",
                        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
                        binary_output=False,
                        return_all_scores=True
                    )
                )
            )
            logger.info("Successfully loaded {}", model)

        self.ignore_topic = ignore_topic

    def pipeline_texts(self, texts: List[str]) -> Dict[int, List[Tuple[int, float]]]:
        sentiment = [
            {i: {SentimentCherryPicker.LABEL_MAPPER[model][label["label"]]: label["score"]for label in labels}
             for i, labels in enumerate(pipeline(texts))}
            for model, pipeline in self.text_classifiers
        ]
        logger.debug("Received {} predictions for the {} texts: {}", len(sentiment), len(texts), sentiment)

        final_scores = defaultdict(lambda: defaultdict(list))
        for text_dict in sentiment:
            for text_number, stance_dict in text_dict.items():
                for stance_label, stance_score in stance_dict.items():
                    final_scores[text_number][stance_label] += [stance_score]

        return {
            text_number: [(score_label, sum(score_list) / len(score_list))
                          for score_label, score_list in final_scores_for_text.items()]
            for text_number, final_scores_for_text in final_scores.items()
        }

    def cherry_pick(self, user: Any, topic: str, stance_probability: float, candidates: List[str]) -> int:
        stance = -1 if stance_probability <= .475 else (1 if stance_probability >= .525 else 0)
        logger.trace(
            "Let's pick the cherry ({} candidates, {})",
            len(candidates),
            SentimentCherryPicker.SCORE_MAPPER[stance]
        )

        if self.ignore_topic:
            topic_sentiment = 0
        else:
            _, topic_sentiment = self.pipeline_texts(texts=[topic]).popitem()
            topic_sentiment.sort(key=lambda e: e[1], reverse=True)
            topic_sentiment = topic_sentiment[0][0]
        expected_sentiment = stance if topic_sentiment == 0 else topic_sentiment * stance
        logger.info(
            "The stance of topic \"{}\" is {}, hence we expect a {} explanation",
            topic, SentimentCherryPicker.SCORE_MAPPER[topic_sentiment],
            SentimentCherryPicker.SCORE_MAPPER[expected_sentiment]
        )

        candidates_sentiments = self.pipeline_texts(texts=candidates)
        candidates_list = []
        for candidate_number, candidate_scores in candidates_sentiments.items():
            if any(map(lambda s: s[0] == expected_sentiment, candidate_scores)):
                candidates_list.append(
                    (candidate_number, [score for label, score in candidate_scores if label == expected_sentiment][0])
                )
                logger.debug("Candidate \"{}\" has a probability of {}% to be {}",
                             candidates[candidate_number], str(round(candidates_list[-1][1]*100)),
                             SentimentCherryPicker.SCORE_MAPPER[expected_sentiment])
            elif expected_sentiment == 0:
                logger.debug("No score available for {}, consider the other scores",
                             SentimentCherryPicker.SCORE_MAPPER[expected_sentiment])
                candidates_list.append(
                    (candidate_number, sum(map(lambda s: s[1], candidate_scores))/len(candidate_scores))
                )
            else:
                logger.warning("Expects a {} sentiment, but only {} are available!",
                               SentimentCherryPicker.SCORE_MAPPER[expected_sentiment],
                               "/".join(map(lambda s: SentimentCherryPicker.SCORE_MAPPER[s[0]], candidate_scores)))

        candidates_list.sort(key=lambda c: c[1], reverse=True)
        return candidates_list[0][0]

    def __str__(self) -> str:
        return "Sentiment-CherryPicker based on {}".format(" and ".join(map(lambda t: t[0], self.text_classifiers)))
