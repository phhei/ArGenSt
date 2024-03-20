from abc import ABC, abstractmethod
from typing import List, Union, Tuple

from loguru import logger

import torch


class AggregatorInterface(torch.nn.Module, ABC):
    """
    An aggregator aggregates several (t_batch, u_batch, features) into two outputs
    - stance: (t_batch, u_batch)
    - encoder-vector for generator: (t_batch, u_batch, encoder_size)
    """

    def __init__(
            self,
            in_features: List[int],
            encoder_size: int,
            binary_stance_label: bool = False,
            normalize_in_vectors: bool = True
    ):
        """
        Initializes an aggregator

        :param in_features: a list of expected feature sizes
        (list of length x => x modules produce an embedding of (topic,user))
        :param encoder_size: the encoder produces an encoded representation of the input (see argument generator) -
        the last dimension must be the same size as this number here
        :param binary_stance_label: should the stance be represented as a number between 0 and 1 or as a softmax-label
        (probability for CON/ PRO)
        :param normalize_in_vectors: flag which actives the normalization of the incoming vectors,
        squeezing all numbers between 0 and 1
        """
        super().__init__()

        if len(in_features) == 0:
            logger.warning("No input features -- can produce only static outputs!")

        self.in_features = in_features
        self.classification_size = 1+int(binary_stance_label)
        self.encoder_size = encoder_size

        self.normalize_in_vectors = normalize_in_vectors

    @abstractmethod
    def forward(self, in_vectors: List[torch.Tensor]) -> Union[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        *[
            in_vector0 (t_batch, u_batch, in_features_0),
            ...
        ] --->
        (t_batch, u_batch, *gen*), (t_batch, u_batch(, *stance*))
        """
        assert len(in_vectors) == len(self.in_features), \
            "Expected {} incoming tensors but got {}".format(len(self.in_features), len(in_vectors))

        logger.trace("Got {} incoming tensors", len(in_vectors))

        if self.normalize_in_vectors:
            ret = []
            logger.trace("Start normalization...")

            for i, num_features in enumerate(self.in_features):
                if num_features <= 2:
                    logger.warning("Too few features to normalize ({}) on the {}. tensor. Apply Sigmoid!",
                                   num_features, i+1)
                    ret.append(torch.sigmoid(in_vectors[i]))
                else:
                    logger.trace("{}. tensor has {} features", i+1, num_features)
                    max_values, _ = torch.max(input=in_vectors[i], dim=-1, keepdim=True)
                    min_values, _ = torch.min(input=in_vectors[i], dim=-1, keepdim=True)

                    ret.append(torch.nan_to_num((in_vectors[i]-min_values)/(max_values-min_values), nan=0.5))
            return ret

        return in_vectors

    def post_process_stance_prediction(self, stance_prediction: torch.Tensor) -> torch.Tensor:
        """
        Post-processed the stance logits to probability [distributions] (in dependence of self.classification_size)
        :param stance_prediction: the logits
        :return: In case of a single probability per topic/ user (self.classification_size==1),
        the logits are map to an interval between 0 and 1 (the last dimension is squeezed),
        else a softmax is applied to the last dimension
        """
        if self.classification_size >= 2:
            return torch.softmax(stance_prediction, dim=-1)

        return torch.squeeze(torch.sigmoid(stance_prediction), dim=-1)

    def __str__(self) -> str:
        return "Aggregator aggregating {} modules: ({})->((*,*,{}), (*,*,{}))".format(
            len(self.in_features),
            "-".join(map(lambda f: "(*,*,{})".format(f), self.in_features)),
            self.encoder_size,
            self.classification_size
        )
