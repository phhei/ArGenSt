import math
import torch

from loguru import logger


class Repeat(torch.nn.Module):
    def __init__(self, desired_vector_length: int):
        super().__init__()
        self.desired_vector_length = desired_vector_length

        logger.debug("Successfully initiate a module which will ensure that all tensors are in shape of (batch, {}) "
                     "by repetition/ truncating", self.desired_vector_length)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        if v.shape[-1] == self.desired_vector_length:
            logger.trace("The input tensor is already in the desired shape (batch, {})", self.desired_vector_length)

        if v.shape[-1] < self.desired_vector_length:
            return torch.repeat_interleave(
                v,
                math.ceil(self.desired_vector_length / v.shape[-1]),
                dim=-1
            ).reshape((v.shape[0], -1))[:, :self.desired_vector_length]

        logger.warning("The input tensor contains {} features but should be cut to {}. {} features will be removed!",
                       v.shape[-1], self.desired_vector_length, v.shape[-1] - self.desired_vector_length)

        return v[:, :self.desired_vector_length]
