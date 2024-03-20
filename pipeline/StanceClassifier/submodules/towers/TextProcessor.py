from pathlib import Path
from typing import Union, Iterable, List, Dict, Optional
from collections import OrderedDict

import torch
import sentence_transformers

from loguru import logger

from pipeline.StanceClassifier.submodules.towers.TowerUtils import Repeat

device = "cuda" if torch.cuda.is_available() else "cpu"


class Linear(torch.nn.Module):
    def __init__(self,
                 root_property_string_to_number_folder: Path,
                 returned_embedding_size: int = 384,
                 normalize_embeddings: bool = False,
                 hidden_layers: Optional[int] = None,
                 activation_function: Optional[torch.nn.Module] = None,
                 dropout_rate: Optional[float] = None,
                 linear_classification_head: bool = True,
                 ignored_properties: Optional[List[str]] = None):
        super().__init__()

        logger.debug("First, have a look into \"{}\"", root_property_string_to_number_folder.absolute())
        self.embeddings: OrderedDict[str, Dict[str, torch.Tensor]] = OrderedDict()
        for file in root_property_string_to_number_folder.glob(pattern="*.tsv"):
            logger.trace("Found property file: {}", file.name)
            lines = file.read_text(encoding="utf-8", errors="ignore").split(sep="\n")
            logger.info("Read {} cases from the file \"{}\"", len(lines), file.stem)

            try:
                # noinspection PyUnboundLocalVariable
                self.embeddings[file.stem] = \
                    {
                        split[0]: torch.tensor(
                            data=[float(s) for s in split[1:]],
                            dtype=torch.float,
                            requires_grad=False,
                            device=device
                        )
                        for line in lines
                        if (not line.startswith("#")) and len(split := line.strip("\t ").split(sep="\t")) >= 2
                    }
                if "default" not in self.embeddings[file.stem]:
                    if len(self.embeddings[file.stem]) >= 1:
                        default_value = torch.mean(torch.stack(self.embeddings[file.stem].values(), dim=0), dim=0)
                        logger.warning("The default value is missing for property \"{}\". Add value {}",
                                       file.stem, default_value)
                    else:
                        logger.opt(exception=True).warning("File \"{}\" seems to be empty! -- default: 0", file.name)
                        default_value = torch.tensor(data=[0], requires_grad=False, device=device)
                    self.embeddings[file.stem]["default"] = default_value
                logger.debug("Successfully have {} cases for property \"{}\"", len(self.embeddings[file.stem]),
                             file.stem)
            except TypeError or RuntimeError:
                logger.opt(exception=True).error("Can't read property \"{}\": {}", file.stem, file)

        logger.success("Successfully loaded the embeddings for {} properties", len(self.embeddings))

        if ignored_properties is not None:
            for ignored_property in ignored_properties:
                try:
                    lost_embeddings = self.embeddings.pop(ignored_property)
                    logger.debug("Successfully removed the property \"{}\": {}",
                                 ignored_property, "/".join(map(lambda kv: ":".join(map(lambda k: str(k),kv)),
                                                                lost_embeddings.items())))
                except KeyError:
                    logger.opt(exception=False).warning(
                        "You want to ignore \"{}\", but this property wasn't loaded anyway", ignored_property
                    )

            logger.info("{} properties left", len(self.embeddings))

        in_features = sum(map(lambda prop: prop["default"].shape[0], self.embeddings.values()))
        logger.debug("There are {} in-features in total: ", in_features)
        logger.trace(" and ".join(map(lambda prop_kv: "{}: {} choices".format(prop_kv[0], len(prop_kv[1])),
                                      self.embeddings.items())))

        if hidden_layers is None:
            logger.trace("No hidden layers")
            self.hidden_layers = torch.nn.Identity()
        else:
            self.hidden_layers = torch.nn.Sequential(
                *[torch.nn.Sequential(
                    torch.nn.Identity() if dropout_rate is None else torch.nn.Dropout(p=dropout_rate),
                    torch.nn.Linear(in_features=in_features, out_features=in_features, bias=True, device=device),
                    torch.nn.Identity() if activation_function is None else activation_function
                ) for _ in range(hidden_layers)]
            )
            logger.debug("Created hidden layers: {}", self.hidden_layers)

        self.head = torch.nn.Sequential(
            torch.nn.Identity() if dropout_rate is None else torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(in_features=in_features, out_features=returned_embedding_size, bias=True, device=device)
            if linear_classification_head else Repeat(desired_vector_length=returned_embedding_size),
            torch.nn.Softmax(dim=1) if normalize_embeddings else torch.nn.Identity()
        )

        logger.success(
            "Successfully created a neural net converting {} input-features to {} output features "
            "(using {} linear layers)",
            in_features, returned_embedding_size, int(linear_classification_head)+(hidden_layers or 0)
        )

    def forward(self, profile_properties: Union[Dict[str, str], List[Dict[str, str]]]) -> torch.FloatTensor:
        """
        Tower for profile information

        :param profile_properties: (a list of) profile property rows
        :return: an encoding of size (batch, returned_embedding_size)
        """
        if isinstance(profile_properties, Dict):
            profile_properties = [profile_properties]
        logger.trace("OK, let's process {} user profiles", len(profile_properties))

        logger.trace("Start embedding")

        embeddings = []

        for profile in profile_properties:
            embedding = []
            for prop, mapper in self.embeddings.items():
                if prop in profile:
                    embedding.append(mapper.get(profile[prop], mapper["default"]))
                    logger.trace("Get \"{}\": {}", prop, embedding[-1])
                else:
                    logger.info("We don't have any information for the property \"{}\" - set default", prop)
                    embedding.append(mapper["default"])
            embeddings.append(torch.concat(tensors=embedding, dim=0))
            logger.trace("Finished embedding for {}: {}", profile, embeddings[-1])
        in_x = torch.stack(tensors=embeddings, dim=0)
        logger.debug("Finished embedding: {}", in_x.shape)

        return self.head(self.hidden_layers(in_x))

    def get_output_dimensionality(self) -> int:
        return self.head[1].out_features

    def __str__(self):
        return "Linear({})->{}d".format(self.head[1].in_features, self.get_output_dimensionality())


class SBERTTower(torch.nn.Module):
    def __init__(self, sbert_model: str = "all-MiniLM-L12-v2", trainable_sbert_model: bool = False,
                 normalize_embeddings: bool = False, ignored_properties: Optional[List[str]] = None):
        super().__init__()

        self.model = sentence_transformers.SentenceTransformer(model_name_or_path=sbert_model, device=device)
        if trainable_sbert_model:
            logger.info("You want to have a trainable SBERT model ({}). This is experimental and may increase "
                        "the training time significantly!", sbert_model)
        else:
            self.model.eval()
        self.trainable_sbert_model = trainable_sbert_model
        self.model.requires_grad_(requires_grad=trainable_sbert_model)
        logger.success("Successfully loaded \"{}\" (up to {} tokens)", sbert_model, self.model.get_max_seq_length())

        self.normalize_embeddings = normalize_embeddings
        self.ignored_properties = ignored_properties

    def forward(self, topic_or_profile_str: Union[str, List[str], List[Iterable[str]]]) -> torch.FloatTensor:
        """
        Tower for text information (can be profile properties, too)

        :param topic_or_profile_str: (al list of) topics, or profile fields
        :return: an encoding of size (batch, sbert_output_size)
        """
        logger.trace("Should embed following strings: {}", topic_or_profile_str)

        if isinstance(topic_or_profile_str, str):
            logger.debug("Only one sample \"{}\" not wrapped in a list - we do that for SBERT now",
                         topic_or_profile_str)
            topic_or_profile_str = [topic_or_profile_str]

        logger.debug("OK, we have to encode {} topics/ profile information", len(topic_or_profile_str))
        if self.trainable_sbert_model:
            logger.trace("Gradients are enabled for the SBERT model (train-set: {}) - so we can train it!",
                         self.model.training)
            embedding = self.model(
                {k: (v.to(device=self.model.device) if isinstance(v, torch.Tensor) else v)
                 for k, v in self.model.tokenize(topic_or_profile_str).items()}
            )["sentence_embedding"]
            if self.normalize_embeddings:
                return torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding

        return self.model.encode(
            sentences=[s if isinstance(s, str) else
                       (" - ".join(
                           [": ".join(kv) for kv in s.items()
                            if self.ignored_properties is None or kv[0] not in self.ignored_properties]
                       ) if isinstance(s, Dict) else
                        " - ".join(s))
                       for s in topic_or_profile_str],
            batch_size=16,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize_embeddings,
            device=device
        )

    def get_output_dimensionality(self) -> int:
        return self.model.get_sentence_embedding_dimension() or 0

    def __str__(self):
        return "SBERT({})->{}d".format(self.model, self.get_output_dimensionality())
