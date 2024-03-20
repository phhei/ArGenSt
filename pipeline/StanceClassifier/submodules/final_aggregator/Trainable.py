from typing import List, Union, Tuple, Optional

import torch

from pipeline.StanceClassifier.submodules.final_aggregator.Interface import AggregatorInterface
from loguru import logger

device = "cuda" if torch.cuda.is_available() else "cpu"


class ShallowNeuralAggregator(AggregatorInterface):
    def __init__(
            self,
            in_features: List[int],
            encoder_size: int,
            binary_stance_label: bool = False,
            normalize_in_vectors: bool = True,
            hidden_layers: Optional[int] = None,
            activation_function: Optional[torch.nn.Module] = None,
            dropout: Optional[float] = None
    ):
        super().__init__(
            in_features=in_features,
            encoder_size=encoder_size,
            binary_stance_label=binary_stance_label,
            normalize_in_vectors=normalize_in_vectors
        )

        self.in_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=sum(self.in_features), out_features=self.encoder_size, bias=True, device=device),
            torch.nn.Identity() if activation_function is None else activation_function,
            torch.nn.Identity() if dropout is None else torch.nn.Dropout(p=dropout)
        )
        logger.info("Created the in-layer: {}", "->".join(map(lambda m: str(m), list(self.in_layer.modules())[1:])))
        if hidden_layers is None:
            self.hidden_layers = torch.nn.Identity()
            logger.debug("No hidden layers created.")
        else:
            self.hidden_layers = torch.nn.Sequential(
                *[torch.nn.Sequential(
                    torch.nn.Linear(
                        in_features=self.encoder_size, out_features=self.encoder_size, bias=True, device=device
                    ),
                    torch.nn.Identity() if activation_function is None else activation_function
                )
                    for _ in range(hidden_layers)]
            )
            logger.info("Created {} hidden layers", hidden_layers)

        self.classification_head_encoder = torch.nn.Linear(
            in_features=self.encoder_size, out_features=self.encoder_size, bias=True, device=device
        )
        self.classification_head_stance = torch.nn.Linear(
            in_features=self.encoder_size, out_features=self.classification_size, bias=True, device=device
        )

        logger.success("Successfully created all layers of the trainable aggregation having {} Dense-layers "
                       "({} parameters in total)",
                       1+(0 if hidden_layers is None else hidden_layers)+2,
                       sum(map(lambda p: torch.numel(p), self.parameters(recurse=True))))

    def forward(self, in_vectors: List[torch.Tensor]) -> Union[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        preprocessed_in_vectors = torch.concat(tensors=super().forward(in_vectors=in_vectors), dim=-1)

        encoded_state_vector = self.hidden_layers(self.in_layer(preprocessed_in_vectors))
        logger.trace("Processed {}topics x {}users using {} layers",
                     preprocessed_in_vectors.shape[0],
                     preprocessed_in_vectors.shape[1], 1+len(list(self.hidden_layers.modules())))

        return \
            self.classification_head_encoder(encoded_state_vector), \
            self.post_process_stance_prediction(self.classification_head_stance(encoded_state_vector))

    def __str__(self) -> str:
        return "{} ({} params)".format(super().__str__(),
                                       sum(map(lambda p: torch.numel(p), self.parameters(recurse=True))))


