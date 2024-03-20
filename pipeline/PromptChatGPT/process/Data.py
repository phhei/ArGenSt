from pathlib import Path
from pandas import read_csv
from typing import Optional, List, Union, Tuple

from pipeline.data_gathering.opinion_arguments.ExternOpinionArgument import ExternOpinionArgument

from dataclasses import dataclass
from loguru import logger


@dataclass
class PromptInstance:
    anchor: ExternOpinionArgument
    pos_instances: List[ExternOpinionArgument]
    neg_instances: Optional[List[ExternOpinionArgument]] = None

    def shots(self, count_neg_instances: bool = True) -> int:
        shots = len(self.pos_instances)
        if count_neg_instances:
            shots += len(self.neg_instances)
        return shots

    def __str__(self) -> str:
        return (f"PromptInstance"
                f"{self.anchor} ({len(self.pos_instances)} pos instances, {len(self.neg_instances)} neg_instances)")


def load_instances(
        csv: Path,
        opinions: List[ExternOpinionArgument],
        min_dimension: int = 0,
        limit_samples: Optional[int] = None
) -> List[PromptInstance]:
    """
    Loads the instances from the prompt csv by pipeline.PromptChatGPT.preprocess with the already loaded opinions.
    :param csv: the path to the prompt csv
    :param opinions: a list of all opinions
    :param min_dimension: how many known profile fields an example-user should have at least
    :param limit_samples: if set, only the first n samples are loaded
    :return: a list of PromptInstances
    """
    if not csv.exists():
        logger.error("{} does not exist, nothing to load", csv.absolute())
        return []

    data_csv = read_csv(filepath_or_buffer=csv, sep=";", header=0, encoding="utf-8", index_col="opinion_argument_id")
    logger.info("Read the data-csv: {}", data_csv.shape)
    data_csv = data_csv[data_csv.minDimension == min_dimension]
    logger.debug("Filtered the data-csv (minDimension: {}): {}", min_dimension, data_csv.shape)

    opinions_dict_by_id = {opinion.argument_id: opinion for opinion in opinions}

    instances = []
    for anchor_id, pos_neg_instances_df in data_csv.groupby(level=0):
        try:
            pos_shots = pos_neg_instances_df[pos_neg_instances_df.positive_or_negative_example == "positive"][
                "similar_topic_ids"
            ][0].replace("\"", "'").strip("[]'").split("', '")
            neg_shots = pos_neg_instances_df[pos_neg_instances_df.positive_or_negative_example == "negative"][
                "similar_topic_ids"
            ][0].replace("\"", "'").strip("[]'").split("', '")
        except IndexError:
            logger.opt(exception=True).warning("Argument {} has no positive or negative examples", anchor_id)
            pos_shots = []
            neg_shots = []

        try:
            instances.append(
                PromptInstance(
                    anchor=opinions_dict_by_id[anchor_id],
                    pos_instances=[opinions_dict_by_id[pos_shot] for pos_shot in pos_shots],
                    neg_instances=[opinions_dict_by_id[neg_shot] for neg_shot in neg_shots]
                )
            )
        except KeyError:
            logger.opt(exception=True).error("Argument {} is corrupt -- skip!", anchor_id)
            continue

        logger.trace("Loaded instance: {}", instances[-1])
        if len(instances) == limit_samples:
            logger.warning("Abort loading instances because limit_samples={} is reached", limit_samples)
            break

    logger.success("Loaded {} instances", len(instances))
    return instances
