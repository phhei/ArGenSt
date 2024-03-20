from pathlib import Path
from typing import Dict, Tuple, List, Any
from pipeline.StanceClassifier.submodules.towers.TextProcessor import SBERTTower, Linear
from pipeline.StanceClassifier.submodules.towers.Graph import SimpleFriendshipGraph
from pipeline.StanceClassifier.submodules.combiners.Interface import CombinerInterface
from pipeline.StanceClassifier.submodules.combiners.Static import MatMulLayer
from pipeline.StanceClassifier.submodules.combiners.Trainable import SimpleNN, DeepCrossNetwork
from pipeline.StanceClassifier.submodules.final_aggregator.Interface import AggregatorInterface
from pipeline.StanceClassifier.submodules.final_aggregator.Trainable import ShallowNeuralAggregator
from pipeline.Utils import convert_dict_types

import torch

from loguru import logger
from forge import fsignature


class StanceClassifier(torch.nn.Module):
    def __init__(self, args: Dict):
        super().__init__()

        logger.debug("Start initializing a StanceClassifier")

        logger.trace("OK, let's initialize the towers")

        self.towers = torch.nn.ModuleDict()
        self.combiners = torch.nn.ModuleList()

        for self_attribute, module_name, keys_to_module_class, default_module_class, default_module_parameters in (
                (
                    self.towers,
                    "topic-text-processor",
                    {
                        "pipeline.StanceClassifier.submodules.towers.TextProcessor.SBERTTower": SBERTTower,
                        "SBERTTower": SBERTTower,
                        "None": None
                    },
                    SBERTTower,
                    dict()
                ),
                (
                    self.towers,
                    "user-id-processor",
                    {
                        "pipeline.StanceClassifier.submodules.towers.Graph.SimpleFriendshipGraph": SimpleFriendshipGraph,
                        "SimpleFriendshipGraph": SimpleFriendshipGraph,
                        "None": None
                    },
                    SimpleFriendshipGraph,
                    {
                        "user_ids": [],
                        "user_id_friendship_relations": [],
                        "user_id_embedding_size": 16
                    }
                ),
                (
                    self.towers,
                    "user-property-processor",
                    {
                        "pipeline.StanceClassifier.submodules.towers.TextProcessor.SBERTTower": SBERTTower,
                        "SBERTTower": SBERTTower,
                        "pipeline.StanceClassifier.submodules.towers.TextProcessor.Linear": Linear,
                        "Linear": Linear,
                        "None": None
                    },
                    Linear,
                    {
                        "root_property_string_to_number_folder": Path(
                            "pipeline/StanceClassifier/submodules/towers/LinearUserProfileEncoderEmbeddings"
                        )
                    }
                ),
                (
                    self.combiners,
                    "MatMul-Combiner",
                    {
                        "pipeline.StanceClassifier.submodules.combiners.Static.MatMulLayer": MatMulLayer,
                        "MatMulLayer": MatMulLayer
                    },
                    None,
                    dict()
                ),
                (
                    self.combiners,
                    "SimpleNN-Combiner-0",
                    {
                        "pipeline.StanceClassifier.submodules.combiners.Trainable.SimpleNN": SimpleNN,
                        "SimpleNN": SimpleNN
                    },
                    None,
                    None
                ),
                (
                        self.combiners,
                        "SimpleNN-Combiner-1",
                        {
                            "pipeline.StanceClassifier.submodules.combiners.Trainable.SimpleNN-additionalOne": SimpleNN,
                            "SimpleNN-additionalOne": SimpleNN
                        },
                        None,
                        None
                ),
                (
                        self.combiners,
                        "DeepCrossNetwork-Combiner-0",
                        {
                            "pipeline.StanceClassifier.submodules.combiners.Trainable.DeepCrossNetwork": DeepCrossNetwork,
                            "DeepCrossNetwork": DeepCrossNetwork
                        },
                        None,
                        None
                ),
                (
                        self.combiners,
                        "DeepCrossNetwork-Combiner-1",
                        {
                            "pipeline.StanceClassifier.submodules.combiners.Trainable.DeepCrossNetwork-additionalOne": DeepCrossNetwork,
                            "DeepCrossNetwork-additionalOne": DeepCrossNetwork
                        },
                        None,
                        None
                )
        ):
            logger.trace("OK, let's initialize the {}", module_name)
            arg_key_intersection = set(keys_to_module_class.keys()).intersection(args.keys())
            if len(arg_key_intersection) == 0:
                logger.warning("You don't define any of these keys ({}), hence, we will initialize the default for {}: "
                               "{}", "/".join(keys_to_module_class.keys()), module_name, default_module_class)
                module_to_add = None if default_module_class is None else \
                    default_module_class(**default_module_parameters)
            else:
                len_arg_key_intersection = len(arg_key_intersection)
                if len_arg_key_intersection >= 2:
                    arg_key_intersection = {arg_key for arg_key in arg_key_intersection
                                            if args[arg_key].get("used_for", module_name) == module_name}
                try:
                    arg_key = arg_key_intersection.pop()
                    if len(arg_key_intersection) >= 1:
                        logger.warning("You defined several keys for {0}: {1}/{2}. We just take \"{2}\"",
                                       module_name, "/".join(arg_key_intersection), arg_key)

                    parameter_dict = args[arg_key].copy()
                    logger.trace(
                        "Found {} parameters for {} (used_for: {})",
                        len(parameter_dict)-1, module_name, parameter_dict.pop("used_for", "not further specified")
                    )
                    if keys_to_module_class[arg_key] is None:
                        module_to_add = None
                        logger.warning("You disabled the architecture part \"{}\" - this is maybe only sensible for a "
                                       "ablation study!", module_name)
                    else:
                        module_to_add = keys_to_module_class[arg_key](
                            **convert_dict_types(kwargs=parameter_dict, signature=fsignature(keys_to_module_class[arg_key]))
                        )
                except KeyError:
                    logger.opt(exception=True).warning("None of the given modules in your config should be used for "
                                                       "\"{}\" (see used_for), but we need at least one!", module_name)
                    module_to_add = None if default_module_class is None else \
                        default_module_class(**default_module_parameters)
            if module_to_add is not None:
                if isinstance(self_attribute, torch.nn.ModuleDict):
                    self_attribute[module_name] = module_to_add
                elif isinstance(self_attribute, torch.nn.ModuleList):
                    self_attribute.append(module_to_add)
                else:
                    self_attribute = module_to_add
                logger.debug("{}: {}", module_name, module_to_add)
                logger.trace("Modules in {}: {}->{}", self_attribute, len(self_attribute)-1, len(self_attribute))

        if len(self.combiners) == 0:
            logger.warning("You must have at least one combiner!")
            self.combiners.append(MatMulLayer())
            logger.info("Created: {}", self.combiners[-1])

        logger.success("Finished initializing all towers ({}) and combiners ({}), "
                       "now, only the classification head is missing", len(self.towers), len(self.combiners))

        self.in_features_final_classification_head = \
            [combiner.get_output_features() for combiner in self.combiners if isinstance(combiner, CombinerInterface)]

        final_aggregator_key = None
        if "pipeline.StanceClassifier.submodules.final_aggregator.Trainable.ShallowNeuralAggregator" in args:
            final_aggregator_key = \
                "pipeline.StanceClassifier.submodules.final_aggregator.Trainable.ShallowNeuralAggregator"
        elif "ShallowNeuralAggregator" in args:
            final_aggregator_key = "ShallowNeuralAggregator"

        if final_aggregator_key is None:
            default_dict = {
                "in_features": self.in_features_final_classification_head,
                "encoder_size": args.get("encoder_size", 768)
            }
            logger.info("No final aggregator defined. Take the default one with: {}", default_dict)

            self.classification_head: AggregatorInterface = ShallowNeuralAggregator(**default_dict)
        else:
            logger.debug("Detect \"{}\" for classification head", final_aggregator_key)

            in_dict = \
                convert_dict_types(kwargs=args[final_aggregator_key], signature=fsignature(ShallowNeuralAggregator))
            if "in_features" in in_dict:
                logger.warning("You overwrite the in-features of the classification head ({}->{}), "
                               "this might cause errors afterwards",
                               self.in_features_final_classification_head, in_dict["in_features"])
                self.in_features_final_classification_head = in_dict["in_features"]
            else:
                logger.debug("\"in_features\" is missing, set {}", self.in_features_final_classification_head)
                in_dict["in_features"] = self.in_features_final_classification_head

            self.classification_head: AggregatorInterface = ShallowNeuralAggregator(**in_dict)

        logger.success("Finished the installation of the stance classifier: ({})->{} combiners->{}",
                       "/".join(self.towers.keys()), len(self.combiners), self.classification_head)

    def forward(self,
                topics: List[str],
                user_ids: List[Any],
                user_properties: List[Dict[str, str]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Processes a forward pass (StanceClassifier)

        :param topics: a batch of topics (claims for which users can have a stance (PRO/CON))
        :param user_ids: a batch of user-ids
        :param user_properties: a batch of user-properties in the form of {user_property: value...}
        :return: a tuple - once a batch of internal states for the generator model (for each topic for each user)
        and a batch of stances. Both Tensors are in shape of (topic_batch, user_batch, *)
        """

        logger.trace("Let's start processing a ({}x{}) batch", len(topics), len(user_ids))

        assert len(user_ids) == len(user_properties)

        topic_embedding = \
            torch.zeros((len(topics), 0), dtype=torch.float, device="cuda" if torch.cuda.is_available() else "cpu") \
            if "topic-text-processor" not in self.towers else self.towers["topic-text-processor"](topics)
        user_embedding = \
            torch.zeros((len(user_ids), 0), dtype=torch.float, device="cuda" if torch.cuda.is_available() else "cpu") \
            if "user-id-processor" not in self.towers else self.towers["user-id-processor"](user_ids)
        user_property_embedding = \
            torch.zeros((len(user_ids), 0), dtype=torch.float, device="cuda" if torch.cuda.is_available() else "cpu") \
            if "user-property-processor" not in self.towers else \
            self.towers["user-property-processor"](user_properties)
        logger.debug("Finished the towers (calculation of embeddings):\n-topic: {}\n-users: {}\n-user-properties: {}",
                     "not calculated" if topic_embedding is None else topic_embedding.shape,
                     "not calculated" if user_embedding is None else user_embedding.shape,
                     "not calculated" if user_property_embedding is None else user_property_embedding.shape)

        final_embeddings = [
            combiner(topic_embedding, user_embedding, user_property_embedding)
            for combiner in self.combiners if isinstance(combiner, CombinerInterface)
        ]
        logger.debug("Successfully calculated {}/{} final embeddings: {}", len(final_embeddings), len(self.combiners),
                     " # ".join(map(lambda e: str(e.shape), final_embeddings)))

        logger.trace("Now the last step with: {}", self.classification_head)
        return self.classification_head(final_embeddings)
