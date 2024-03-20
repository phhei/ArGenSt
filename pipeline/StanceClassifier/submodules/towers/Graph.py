from typing import List, Any, Optional, Iterable, Tuple, Callable, Union

import torch

from loguru import logger

device = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleFriendshipGraph(torch.nn.Module):
    """
    A GraphNN processing an user-id (converting to a user-id-embedding)

    OUTPUTS (u_batch, user_id_embedding_size)
    """
    def __init__(
            self,
            user_ids: List[Any],
            user_id_friendship_relations: Iterable[Tuple[Any, Any, float]],
            user_id_embedding_size: int,
            reserve_slots_for_unknown_users: Optional[int] = 100,
            enable_edge_weighting: bool = False,
            user_initial_node_embeddings: Optional[List[torch.Tensor]] = None,
            aggregation_function_node_embeddings: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
            aggregation_processing_module: Optional[torch.nn.Module] = None
    ):
        logger.debug("OK, creating a neural net for {} users", len(user_ids))

        super().__init__()

        self.user_ids = user_ids
        self.map_user_id_to_position = {u: i for i, u in enumerate(self.user_ids)}
        self.remaining_free_slots = reserve_slots_for_unknown_users

        self.user_id_embedding_size = user_id_embedding_size
        if user_initial_node_embeddings is None:
            self.user_id_embeddings = torch.nn.Parameter(
                data=torch.rand((len(user_ids)+(reserve_slots_for_unknown_users or 0), user_id_embedding_size),
                                device=device),
                requires_grad=True
            )
        elif len(user_initial_node_embeddings) == len(self.user_ids):
            if reserve_slots_for_unknown_users is not None:
                logger.info("Now we have to ensure embeddings for unknown users, too. Hence, we add {} empty vectors",
                            reserve_slots_for_unknown_users)
                user_initial_node_embeddings.extend(
                    [torch.rand_like(user_initial_node_embeddings[0])] * reserve_slots_for_unknown_users
                )
            self.user_id_embeddings = torch.nn.Parameter(
                data=torch.stack(tensors=user_initial_node_embeddings, dim=0).to(device),
                requires_grad=True
            )
        else:
            logger.error("You have {} users but provide {} initial user embeddings! We ignore them!",
                         len(self.user_ids), len(user_initial_node_embeddings))
            self.user_id_embeddings = torch.nn.Parameter(
                data=torch.ones((len(user_ids)+(reserve_slots_for_unknown_users or 0), user_id_embedding_size),
                                 device=device),
                requires_grad=True
            )

        self.friendship_matrix = torch.diag(torch.ones((len(user_ids)+(reserve_slots_for_unknown_users or 0),),
                                                       device=device))
        for friendship in user_id_friendship_relations:
            user_id_1 = friendship[0]
            user_id_2 = friendship[1]
            friendship_strength = friendship[2] if len(friendship) >= 3 else 1.
            try:
                self.friendship_matrix[self.map_user_id_to_position[user_id_1],
                                       self.map_user_id_to_position[user_id_2]] = friendship_strength
                self.friendship_matrix[self.map_user_id_to_position[user_id_2],
                                       self.map_user_id_to_position[user_id_1]] = friendship_strength
                logger.trace("Added a edge between \"{}\" and \"{}\"", user_id_1, user_id_2)
            except KeyError:
                logger.opt(exception=True).warning("Ignore entry \"{}\"<->\"{}\"", user_id_1, user_id_2)
            except RuntimeError:
                logger.opt(exception=True).critical("Can't setup the friendship matrix")

        if enable_edge_weighting:
            logger.info("We make the {0}x{0}-friendship matrix trainable now, adding {1} ({2}‰ trainable) params",
                        len(self.user_ids), torch.numel(self.friendship_matrix),
                        str(round(1000*(torch.sum(self.friendship_matrix)/torch.numel(self.friendship_matrix)).item(),
                                  1)))
            self.friendship_matrix = torch.nn.Parameter(data=self.friendship_matrix, requires_grad=True)
        else:
            logger.debug("Created a friendship matrix with {} fields", torch.numel(self.friendship_matrix))

        if aggregation_function_node_embeddings is None:
            logger.info("We have to aggregate all the node embeddings representing the neighbourhood of a node. "
                        "You don't specify one, so let's just sum the embeddings")
            self.aggregation_function_node_embeddings = torch.sum
        else:
            self.aggregation_function_node_embeddings = aggregation_function_node_embeddings

        if aggregation_processing_module is None:
            logger.info("You don't define any module processing the aggregated node embeddings for a final node "
                        "representation. Hence, we use a simple dense-layer here.")
            self.aggregation_processing_module = torch.nn.Linear(
                in_features=user_id_embedding_size, out_features=user_id_embedding_size, bias=True, device=device
            )
        else:
            self.aggregation_processing_module = aggregation_processing_module

        logger.success("Successfully initialized the graph neural net having {} parameters, representing {} users",
                       sum(map(lambda p: torch.numel(p), self.parameters(recurse=True))),
                       len(self.user_ids))

    def forward(self, user_ids: Union[torch.Tensor, List[Any]]) -> torch.Tensor:
        """
        Receiving a list of user-ids (can be strings, too),
        outputting a encoded (u_batch, user_id_embedding_size)-tensor // forward GNN-pass
        :param user_ids: user-ids (can be strings, too)
        :return: (u_batch, user_id_embedding_size)-tensor
        """
        if isinstance(user_ids, torch.Tensor):
            already_position_encoded = True
            logger.warning("You're inputting a tensor ({}) instead of user-ids, hence we assume that your tensor is "
                           "already aware of the position of the specific users in self.user_ids", user_ids)
            user_ids = user_ids.type(torch.LongTensor)
            if len(user_ids.shape) == 2:
                user_ids = user_ids[:, 0]
        else:
            already_position_encoded = False

        ret = []

        for user_id in user_ids:
            try:
                ret.append(self._compute_single_user(user_id, already_position_encoded))
            except (KeyError, IndexError):
                logger.opt(exception=True).warning("User \"{}\" is unknown (was not in the friends-database). "
                                                   "Hence, we can say anything about him/ her", user_id)
                if self.remaining_free_slots is None:
                    ret.append(torch.rand((self.user_id_embedding_size, ), device=device))
                else:
                    logger.info("Let's try to add \"{}\" to the friendship graph (without any friends)", user_id)
                    if self.remaining_free_slots <= 0:
                        logger.error("We have no free slots for unknown users left! ({} full)", len(self.user_ids))
                        ret.append(torch.zeros((self.user_id_embedding_size,), device=device))
                    else:
                        logger.debug("We have {} free slots left, so we add \"{}\" to the friendship graph",
                                     self.remaining_free_slots, user_id)
                        self.user_ids.append(user_id)
                        self.map_user_id_to_position[user_id] = len(self.user_ids)-1
                        self.remaining_free_slots -= 1
                        ret.append(self._compute_single_user(user_id, already_position_encoded))
            except RuntimeError:
                logger.opt(exception=True).error("Can't process user \"{}\"", user_id)
                ret.append(torch.zeros((self.user_id_embedding_size,), device=device))

        logger.debug("Successfully processed {} users", len(ret))

        return torch.stack(tensors=ret, dim=0)

    def _compute_single_user(self, user_id: Union[Any, torch.Tensor], already_position_encoded: bool) -> torch.Tensor:
        if already_position_encoded:
            user_id_position = user_id.item()
        else:
            user_id_position = self.map_user_id_to_position[user_id]

        friends_tensor = self.friendship_matrix[user_id_position]
        logger.trace("The user \"{}\" is at position {}, having {} fiends (including himself)",
                     user_id, user_id_position, torch.count_nonzero(friends_tensor).item())

        masked_node_embeddings = \
            torch.permute(torch.permute(self.user_id_embeddings, dims=(1, 0)) * friends_tensor, dims=(1, 0))

        aggregated_node_embeddings = self.aggregation_function_node_embeddings(masked_node_embeddings, 0)
        logger.trace("Aggregated node embeddings: {}", aggregated_node_embeddings)

        return self.aggregation_processing_module(aggregated_node_embeddings)

    def get_output_dimensionality(self) -> int:
        return self.user_id_embedding_size

    def __str__(self):
        return \
            "SimpleFriendshipGraph(Tower processing the {} users (excluding their properties, " \
            "including their friendships)): (R{}R --> {}) --> {} ".format(
                len(self.user_ids),
                torch.count_nonzero(self.friendship_matrix),
                self.aggregation_function_node_embeddings,
                self.aggregation_processing_module
            )
