from typing import Callable

import torch

from loguru import logger
from pipeline.StanceClassifier.submodules.combiners.Interface import CombinerInterface


def d3_matmul(x1: torch.Tensor, x2: torch.Tensor, merge_3d: bool) -> torch.Tensor:
    assert len(x1.shape) == len(x2.shape) == 3, "can only process 3d-tensors"
    if merge_3d:
        assert x1.shape[0] == x2.shape[0], "Both 3d-matrix have to be the same size in first shape position"
        return torch.stack(tensors=[torch.matmul(x1[i], x2[i]) for i in range(x1.shape[0])], dim=0)
    else:
        return torch.stack(
            tensors=[torch.stack(tensors=[torch.matmul(x1_sl, x2_sl) for x2_sl in x2], dim=0) for x1_sl in x1],
            dim=0
        )


class MatMulLayer(CombinerInterface):
    """
    A combiner combining the tower-processed topic-embedding, the tower-processed user-id-embedding
    and tower-processed user-profile-embedding

    OUTPUTS (t_batch, u_batch, 1)
    """
    def __init__(
            self,
            matmul_topic_user_id_pooling: Callable[[torch.Tensor, int], torch.Tensor] = torch.mean,
            matmul_user: Callable[[torch.Tensor, int], torch.Tensor] = torch.mean
    ):
        super().__init__()

        self.matmul_topic_user_id_pooling = matmul_topic_user_id_pooling
        self.matmul_user = matmul_user

    def forward(self, topic_embedding: torch.FloatTensor,
                user_id_embedding: torch.FloatTensor,
                user_profile_embedding: torch.FloatTensor) -> torch.FloatTensor:
        """
        [
            topic_embedding: (t_batch, t_embedding),
            user_id_embedding: (u_batch, u_embedding),
            user_profile_embedding: (u_batch, #features*up_embedding)
        ] --->
        (t_batch, u_batch, 1)
        """
        if any(map(lambda e: e.shape[-1] == 0, (topic_embedding, user_id_embedding, user_profile_embedding))):
            raise NotImplementedError("MatMulLayer doesn't support empty embeddings "
                                      "(use another combiner for ablation study)")

        if len(topic_embedding.shape) == 2:
            logger.warning("We have to extend the embedding of the topic (({0})->({0},1))", topic_embedding.shape)
            topic_embedding = torch.unsqueeze(topic_embedding, dim=-1)
        if len(user_id_embedding.shape) == 3:
            logger.debug("We have to squeeze the embedding of the user ({}) to unsqueeze the vector dynamically)",
                         user_id_embedding.shape)
        else:
            user_id_embedding = torch.unsqueeze(user_id_embedding, dim=-1)
        user_id_embedding_squeezed = user_id_embedding[:, :, 0]
        if len(user_profile_embedding.shape) == 3:
            logger.debug("We have to squeeze the embedding of the user ({})", user_profile_embedding.shape)
            user_profile_embedding = user_profile_embedding[:, :, 0]

        matmul_topic_user_id = d3_matmul(x1=topic_embedding,
                                         x2=torch.unsqueeze(user_id_embedding_squeezed, dim=1),
                                         merge_3d=False)
        pooled_topic_user = self.matmul_topic_user_id_pooling(matmul_topic_user_id, -2)
        logger.trace(
            "Processed \"matmul_topic_user_id\": {} --> {}",
            matmul_topic_user_id.shape, pooled_topic_user.shape
        )
        matmul_user = d3_matmul(x1=user_id_embedding, x2=torch.unsqueeze(user_profile_embedding, dim=1), merge_3d=True)
        pooled_user = self.matmul_user(matmul_user, -1)
        logger.trace(
            "Processed \"matmul_user\": {} --> {}",
            matmul_user.shape, pooled_user.shape
        )

        return torch.unsqueeze(
            input=torch.stack(
                tensors=[
                    torch.stack(
                        tensors=[torch.dot(tu_sl[u], pooled_user[u]) for u in range(pooled_user.shape[0])],
                        dim=0
                    )
                    for tu_sl in pooled_topic_user
                ],
                dim=0
            ),
            dim=-1
        )

    def get_output_features(self) -> int:
        return 1
