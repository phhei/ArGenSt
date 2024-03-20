from typing import Optional, Literal

import torch

from loguru import logger
from pipeline.StanceClassifier.submodules.combiners.Interface import CombinerInterface

device = "cuda" if torch.cuda.is_available() else "cpu"


def concat_vectors(topic_embedding: torch.FloatTensor,
                   user_id_embedding: torch.FloatTensor,
                   user_profile_embedding: torch.FloatTensor) -> torch.FloatTensor:
    """
    [
        topic_embedding: (t_batch, t_embedding),
        user_id_embedding: (u_batch, u_embedding),
        user_profile_embedding: (u_batch, #features*up_embedding)
    ] --->
    (t_batch, u_batch, t_embedding+u_embedding+#features*up_embedding)
    """
    squeezed_input_tensors = \
        [t[:, :, 0] if len(t.shape) == 3 else t
         for t in (topic_embedding, user_id_embedding, user_profile_embedding)]
    topic_batch_size = topic_embedding.shape[0]
    user_batch_size = user_id_embedding.shape[0]
    logger.trace("topic_batch_size: {}/ user_batch_size: {}", topic_batch_size, user_batch_size)

    concatenation = torch.concat(
        tensors=(
            torch.unsqueeze(squeezed_input_tensors[0], dim=1).repeat(1, user_batch_size, 1),
            torch.unsqueeze(squeezed_input_tensors[1], dim=0).repeat(topic_batch_size, 1, 1),
            torch.unsqueeze(squeezed_input_tensors[2], dim=0).repeat(topic_batch_size, 1, 1)
        ),
        dim=-1
    )
    logger.debug("Concatenate all inputs to a matrix of shape {}", concatenation.shape)

    return concatenation


class SimpleNN(CombinerInterface):
    """
    A combiner combining the tower-processed topic-embedding, the tower-processed user-id-embedding
    and tower-processed user-profile-embedding in a simple neural way

    OUTPUTS (t_batch, u_batch, fuse_embedding_size)
    """
    def __init__(
            self,
            topic_embedding_size: int,
            user_id_embedding_size: int,
            user_profile_embedding_size: int,
            fuse_embedding_size: int,
            hidden_layers: Optional[int] = None,
            dropout: Optional[float] = None,
            activation_function_in_hidden_layers: Optional[torch.nn.Module] = None
    ):
        super().__init__()

        self.in_features = topic_embedding_size+user_id_embedding_size+user_profile_embedding_size
        self.out_features = fuse_embedding_size

        if hidden_layers is None:
            self.hidden_layers = torch.nn.Identity()
        else:
            self.hidden_layers = torch.nn.Sequential(
                torch.nn.Identity() if dropout is None else torch.nn.Dropout(p=dropout),
                *[torch.nn.Sequential(
                    torch.nn.Linear(
                        in_features=self.in_features, out_features=self.in_features, bias=True, device=device
                    ),
                    torch.nn.Identity()
                    if activation_function_in_hidden_layers is None else activation_function_in_hidden_layers
                ) for _ in range(hidden_layers)]
            )
        self.dropout = torch.nn.Identity() if dropout is None else torch.nn.Dropout(p=dropout)
        self.embedding_head = torch.nn.Linear(
            in_features=self.in_features, out_features=self.out_features, bias=True, device=device
        )

    def forward(self, topic_embedding: torch.FloatTensor,
                user_id_embedding: torch.FloatTensor,
                user_profile_embedding: torch.FloatTensor) -> torch.FloatTensor:
        """
        [
            topic_embedding: (t_batch, t_embedding),
            user_id_embedding: (u_batch, u_embedding),
            user_profile_embedding: (u_batch, #features*up_embedding)
        ] --->
        (t_batch, u_batch, fuse_embedding_size)
        """
        return self.embedding_head(
            self.dropout(
                self.hidden_layers(
                    concat_vectors(topic_embedding=topic_embedding,
                                   user_id_embedding=user_id_embedding,
                                   user_profile_embedding=user_profile_embedding)
                )
            )
        )

    def get_output_features(self) -> int:
        return self.out_features


class DeepCrossNetwork(CombinerInterface):
    """
    A combiner combining the tower-processed topic-embedding, the tower-processed user-id-embedding
    and tower-processed user-profile-embedding in the neural way of

    DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems
    by Ruoxi Wang, Rakesh Shivanna, Derek Z. Cheng, Sagar Jain, Dong Lin, Lichan Hong, Ed H. Chi

    OUTPUTS (t_batch, u_batch, fuse_embedding_size)
    """

    class CrossModule(torch.nn.Module):
        def __init__(self, features: int):
            super().__init__()

            self.linear_core = torch.nn.Linear(in_features=features, out_features=features, bias=True, device=device)

        def forward(self, x_in: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:
            return (x_0 * self.linear_core(x_in)) + x_in

    def __init__(
            self,
            topic_embedding_size: int,
            user_id_embedding_size: int,
            user_profile_embedding_size: int,
            fuse_embedding_size: int,
            form: Literal["stacked", "parallel"] = "stacked",
            num_layers: int = 3,
            activation_function: Optional[torch.nn.Module] = None,
            dense_feature_appendix: Optional[int] = None
    ):
        super().__init__()

        self.in_features = topic_embedding_size + user_id_embedding_size + user_profile_embedding_size
        self.out_features = fuse_embedding_size

        self.form = form

        self.in_embedding_appendix_module = \
            None if dense_feature_appendix is None else \
                torch.nn.Linear(in_features=self.in_features,
                                out_features=dense_feature_appendix,
                                bias=False,
                                device=device)

        self.cross_layers = torch.nn.ModuleList(
            [DeepCrossNetwork.CrossModule(features=self.in_features + (dense_feature_appendix or 0)) for _ in range(num_layers)]
        )
        self.linear_layers = torch.nn.Sequential(
            *[torch.nn.Sequential(
                torch.nn.Linear(in_features=self.in_features + (dense_feature_appendix or 0),
                                out_features=self.in_features + (dense_feature_appendix or 0),
                                bias=True,
                                device=device),
                torch.nn.Identity() if activation_function is None else activation_function
            ) for _ in range(num_layers)]
        )

        self.embedding_head = torch.nn.Linear(
            in_features=(self.in_features + (dense_feature_appendix or 0))*(1+int(form == "parallel")),
            out_features=self.out_features,
            bias=True,
            device=device
        )

    def forward(self, topic_embedding: torch.FloatTensor,
                user_id_embedding: torch.FloatTensor,
                user_profile_embedding: torch.FloatTensor) -> torch.FloatTensor:
        """
        [
            topic_embedding: (t_batch, t_embedding),
            user_id_embedding: (u_batch, u_embedding),
            user_profile_embedding: (u_batch, #features*up_embedding)
        ] --->
        (t_batch, u_batch, fuse_embedding_size)
        """
        x_0 = concat_vectors(
            topic_embedding=topic_embedding,
            user_id_embedding=user_id_embedding,
            user_profile_embedding=user_profile_embedding
        )

        if self.in_embedding_appendix_module is not None:
            logger.trace("We have to calculate the dense appendix first ({} features)",
                         self.in_embedding_appendix_module.out_features)
            x_0 = torch.concat(
                tensors=(x_0, self.in_embedding_appendix_module(x_0)),
                dim=-1
            )
            logger.trace("DONE, each in-vector has {} features now", x_0.shape[-1])

        cross_embedding = x_0
        for cross_layer in self.cross_layers:
            cross_embedding = cross_layer(x_in=cross_embedding, x_0=x_0)

        if self.form == "stacked":
            logger.trace("Select form \"{}\", setting the linear layers on top", self.form)
            return self.embedding_head(self.linear_layers(cross_embedding))

        logger.trace("Select form \"{}\", so we have to calculate the {} linear layers in parallel",
                     self.form, len(list(self.linear_layers.modules())))
        linear_embedding = self.linear_layers(x_0)
        return self.embedding_head(torch.concatenate(tensors=(cross_embedding, linear_embedding), dim=-1))

    def get_output_features(self) -> int:
        return self.out_features
