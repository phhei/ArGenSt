import torch
import abc


class CombinerInterface(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, topic_embedding: torch.FloatTensor,
                user_id_embedding: torch.FloatTensor,
                user_profile_embedding: torch.FloatTensor) -> torch.FloatTensor:
        pass

    @abc.abstractmethod
    def get_output_features(self) -> int:
        """
        Function to fetch the size of the last dimension of the returned tensor from the forward-function of this module

        :return: number of output-features
        """
        pass
