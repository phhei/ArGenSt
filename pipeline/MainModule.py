from typing import Dict, List, Any, Union, Tuple, Optional
from itertools import chain

import numpy
import torch
import json

from loguru import logger
from transformers import T5Tokenizer, BatchEncoding, GenerationConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.utils import ModelOutput

from pipeline.StanceClassifier.MainModule import StanceClassifier
from pipeline.ArgumentGenerator.TransformerEncoderDecoder import T5ForConditionalArgGeneration


class StanceArgumentGeneratorModule(torch.nn.Module):
    def __init__(self, args: Dict):
        super().__init__()

        logger.debug("Initialize the overall module for stance AND Argument generator with {} different "
                     "overall-parameters", len(args))

        try:
            t5_model = args.pop("argument_generator_module")
        except KeyError:
            try:
                t5_model = args.pop("t5")
            except KeyError:
                logger.opt(exception=True).warning("You don't define any args for your argument generator module! "
                                                   "You can define the T5-model by giving a string using the "
                                                   "\"argument_generator_module\"-key!")
                t5_model = "t5-small"

        self.argument_module: T5ForConditionalArgGeneration = T5ForConditionalArgGeneration.from_pretrained(
            pretrained_model_name_or_path=t5_model,
            return_dict=True,
            max_length=150,
            min_length=15,
            do_sample=True
        )
        logger.success("Successfully initialized the Argument Generator: {}", self.argument_module)
        if "load_tokenizer" not in args or str(args.pop("load_tokenizer")).upper() == "TRUE":
            self.argument_tokenizer: Optional[T5Tokenizer] = \
                T5Tokenizer.from_pretrained(pretrained_model_name_or_path=t5_model)
            logger.info("Successfully load the tokenizer, too: {}", self.argument_tokenizer)
            self.forward_user_properties_into_argument_generator = \
                args.pop("forward_user_properties_into_argument_generator", "False").upper() == "TRUE"
        else:
            self.argument_tokenizer: Optional[T5Tokenizer] = None

        try:
            stance_module_parameters = args.pop("stance_module")
        except KeyError:
            stance_module_parameters = dict()
        if len(stance_module_parameters) == 0:
            logger.warning("You don't define any args for your stance classification module! "
                           "Please consider \"stance_module\" as key!")
        elif not isinstance(stance_module_parameters, Dict):
            logger.error("Malformed config-file. Expected a dict as \"stance_module\", but got {} -- ignore!",
                         type(stance_module_parameters))
            stance_module_parameters = dict()
        stance_module_parameters["encoder_size"] = self.argument_module.config.d_model
        logger.debug("Set \"stance_module_parameters\": {}", stance_module_parameters["encoder_size"])

        self.stance_module = StanceClassifier(args=stance_module_parameters)
        logger.success("Successfully initialized the StanceClassifier: {}", self.stance_module)

        logger.info("We're done, we don't used following args: {}", "/".join(map(lambda k: str(k), args.keys())))

    def _tokenize_topic(
            self,
            topics: Union[List[str], Tuple[List[str], BatchEncoding]],
            user_properties: List[Dict[str, str]],
            arguments_per_topic_per_user: Optional[Union[List[List[str]], torch.LongTensor]]
    ) -> Tuple[List[str], BatchEncoding]:
        if isinstance(topics, Tuple):
            return topics

        logger.trace("We have to tokenize the topics first for our argument generator")
        if self.argument_tokenizer is None:
            raise AttributeError("No tokenizer loaded, please use tuples for \"topics\"!")
        topics_raw = topics
        topics_tokenized = self.argument_tokenizer(
            text=list(chain.from_iterable([[topic] * len(user_properties) for topic in topics_raw])),
            text_pair=list(chain.from_iterable([[" and ".join(
                [f"{prop.replace('birthday', 'age').replace('_', ' ')} is {str_v.lower()}"
                 for prop, v in user.items()
                 if (str_v := str(v)) not in ("0", "Not Saying", "Prefer not to say", "- Private -", "Other")
                 and prop != "num_of_not_null_entries"]
            ) for user in user_properties]] * len(topics_raw)))
            if self.forward_user_properties_into_argument_generator else None,
            text_target=list(chain.from_iterable(arguments_per_topic_per_user))
            if arguments_per_topic_per_user is not None and isinstance(arguments_per_topic_per_user, List)
            else None,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
            is_split_into_words=False
        )

        return topics_raw, topics_tokenized.to(device="cuda" if torch.cuda.is_available() else "cpu")

    def forward(
            self,
            topics: Union[List[str], Tuple[List[str], BatchEncoding]],
            user_ids: List[Any],
            user_properties: List[Dict[str, str]],
            arguments_per_topic_per_user: Optional[Union[List[List[str]], torch.LongTensor]],
            encoder_output_shift_impact: float = 1.
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, Optional[torch.Tensor]]]:
        """
        Performs a forward-pass
        :param topics: a list of all topics which should be discussed (or a string-list and an already performed batch
        encoding to save time)
        :param user_ids: a list of all user ids which should sit in the jury (should give their opinions)
        :param user_properties: the profiles of thw users
        :param arguments_per_topic_per_user: to get a loss/ have a proper learning of the argument generator:
        a topic-length-list containing the target opinions of all jury-members (users)
        :param encoder_output_shift_impact: How much impact should the encoder_output_shift have.
        Default is 1 which means a simple addition of both original encoder hidden states and the shift. Here, you can
        scale the shift. BE AWARE: if the model is not used to incooperate such a shifting, the generation results will
        be odd. You should start fine-tuning with a decreased impact vector as .1
        :return: the stance probabilities and the output of the argument generator (in addition to the calculated loss
        of the argument generator).
        Hence, shape-ly speaken:
        ([topic_batch, user_batch(, 2)], ([topic_batch, user_batch, sequence_length, vocab_length], 1))
        """
        logger.debug("Got {} topics and {} user-ids. Hence, we have to do {} predictions",
                     lt := len(topics[0]) if isinstance(topics, Tuple) else len(topics),
                     len(user_ids), lt*len(user_ids))

        topics_raw, topics_tokenized = self._tokenize_topic(
            topics=topics, user_properties=user_properties, arguments_per_topic_per_user=arguments_per_topic_per_user
        )

        if arguments_per_topic_per_user is not None and isinstance(arguments_per_topic_per_user, torch.Tensor):
            logger.debug("You already tokenized the argument text target (not recommended),"
                         "so we have to add them to the argument generator batch // {}",
                         arguments_per_topic_per_user.shape)
            topics_tokenized["labels"] = arguments_per_topic_per_user

        logger.debug("OK, now compute the stance...")

        encoder_shift_generator_model, stances_probabilities = self.stance_module(
            topics=topics_raw,
            user_ids=user_ids,
            user_properties=user_properties
        )
        logger.info("Got all from the stance module ({}/{})",
                    encoder_shift_generator_model.shape, stances_probabilities.shape)

        logger.trace("Now, let's ask our argument generator!")

        argument_model_output: Seq2SeqLMOutput = self.argument_module(
            **topics_tokenized,
            encoder_output_shift=torch.unsqueeze(
                torch.flatten(encoder_shift_generator_model, start_dim=0, end_dim=-2), dim=1
            ).repeat((1, topics_tokenized.get("input_ids").shape[-1], 1)),
            encoder_output_shift_impact=encoder_output_shift_impact
        )
        logger.info("Got the predictions of the argument model (with loss {})",
                    "NaN" if argument_model_output.loss is None else argument_model_output.loss.item())

        logger.trace("Forward pass done, clean up...")

        return stances_probabilities, \
               (torch.reshape(argument_model_output.logits,
                              (len(topics_raw), len(user_ids), *argument_model_output.logits.shape[-2:])),
                argument_model_output.loss)

    def inference(
            self,
            topics: Union[List[str], Tuple[List[str], BatchEncoding]],
            user_ids: List[Any],
            user_properties: List[Dict[str, str]],
            encoder_output_shift_impact: float = 1.,
            args_for_generate_function: Optional[Dict[str, Any]] = None,
            return_config_for_generate_function: bool = True
    ) -> Dict[str, Dict[Any, Tuple[Union[float, numpy.ndarray], Optional[Union[numpy.ndarray, List[str]]]]]]:
        """
        Does an inference step (test case) - not suitable for training (use the forward-function for training)
        :param topics: a list of all topics which should be discussed (or a string-list and an already performed batch
        encoding to save time)
        :param user_ids: a list of all user ids which should sit in the jury (should give their opinions)
        :param user_properties: the profiles of the users
        :param encoder_output_shift_impact: How much impact should the encoder_output_shift have.
        Default is 1 which means a simple addition of both original encoder hidden states and the shift. Here, you can
        scale the shift. BE AWARE: if the model is not used to incooperate such a shifting, the generation results will
        be odd. You should start fine-tuning with a decreased impact vector as .1
        :param args_for_generate_function: Arguments for the text-generation-method (in a dict-format). See
        https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/text_generation#transformers.GenerationConfig
        :param return_config_for_generate_function: if set to true (default) the returned dictionary will also contain
        all args that were used to generate the argument strings (key "gen_kwargs")
        :return: a dictionary {topic1, {user1: (stance, generated-argument(s)), user2....}, topic2...}.
        Note: depending on your stance-configs, the stance may be expressed by a single probability (between 0 and 1)
        or as a tuple of two probabilities (CON-prob, PRO-prob).
        Note: depending on your args_for_generate_function, the generated-argument(s) are only one or several
        (for "cherry-picking"). If you refuse to set a tokenizer, only the logits are returned, not text!
        Note: if you set \"skip_generation = True\" in args_for_generate_function, the generation process is skipped.
        Therefore, the second tuple will be None
        """
        topics_raw, topics_tokenized = self._tokenize_topic(
            topics=topics, user_properties=user_properties, arguments_per_topic_per_user=None
        )

        logger.debug("Got {} topics ({}) and {} users",
                     len(topics_raw), ", ".join(map(lambda t: "\"{}\"".format(t), topics_raw)), len(user_ids))

        logger.debug("OK, now compute the stance...")

        ret = {}

        with torch.no_grad():
            encoder_shift_generator_model, stances_probabilities = self.stance_module(
                topics=topics_raw,
                user_ids=user_ids,
                user_properties=user_properties
            )
            logger.info("Got all from the stance module ({}/{})",
                        encoder_shift_generator_model.shape, stances_probabilities.shape)
            numpy_stance_probabilities = stances_probabilities.cpu().numpy()

            if "skip_generation" in args_for_generate_function and args_for_generate_function["skip_generation"]:
                logger.warning("Skip generation...")
                numpy_generated_encoded_sequences = None
            else:
                logger.trace("Now, let's ask our argument generator!")

                generation_config = GenerationConfig.from_model_config(model_config=self.argument_module.config)
                if args_for_generate_function is None:
                    args_for_generate_function = {"return_dict_in_generate": True}
                else:
                    args_for_generate_function["return_dict_in_generate"] = True
                unused_args = generation_config.update(**args_for_generate_function)
                if len(unused_args) >= 1:
                    logger.warning("{} of your \"args_for_generate_function\" were unexpected and, hence, ignored: {}",
                                   len(unused_args),
                                   " and ".join(map(lambda kv: "{} ({})".format(*kv), unused_args.items())))
                else:
                    logger.info("Successfully defined {} additional args for the generation",
                                len(args_for_generate_function))

                logger.trace("Use the following special generation-args: {}",
                             generation_config.to_json_string(use_diff=True))
                if return_config_for_generate_function:
                    ret["gen_kwargs"] = json.loads(s=generation_config.to_json_string(use_diff=False))

                generated_encoded_outputs: ModelOutput = self.argument_module.generate(
                    **topics_tokenized,
                    encoder_output_shift=torch.unsqueeze(
                        torch.flatten(encoder_shift_generator_model, start_dim=0, end_dim=-2), dim=1
                    ).repeat((1, topics_tokenized.get("input_ids").shape[-1], 1)),
                    encoder_output_shift_impact=encoder_output_shift_impact,
                    generation_config=generation_config
                )

                logger.trace("Generation done, got the following keys: {}",
                             ", ".join(map(lambda k: str(k), generated_encoded_outputs.keys())))

                if "sequences" not in generated_encoded_outputs:
                    logger.warning("Unexpected model output (type {}) (sequences are missing)",
                                   type(generated_encoded_outputs))
                    numpy_generated_encoded_sequences = None
                else:
                    # sequences (torch.LongTensor of shape (batch_size*num_return_sequences, sequence_length)) —
                    # The generated sequences. The second dimension (sequence_length) is either equal to max_length or
                    # shorter if all batches finished early due to the eos_token_id.
                    generated_encoded_sequences: torch.LongTensor = \
                        torch.reshape(generated_encoded_outputs["sequences"],
                                      (len(topics_raw), len(user_ids), generation_config.num_return_sequences, -1))
                    logger.success("Successfully got the generated sequences of shape {}",
                                   generated_encoded_sequences.shape)
                    numpy_generated_encoded_sequences = generated_encoded_sequences.cpu().numpy()

        logger.info("The final step: aggregate all the results!")
        for topic_index, topic in enumerate(topics_raw):
            logger.trace("Process {}.: {}", topic_index, topic)
            ret[topic] = dict()
            for user_index, user_id in enumerate(user_ids):
                ret[topic][user_id] = (
                    numpy_stance_probabilities[topic_index][user_index],
                    (numpy_generated_encoded_sequences[topic_index][user_index] if self.argument_tokenizer is None else
                     self.argument_tokenizer.batch_decode(
                        sequences=numpy_generated_encoded_sequences[topic_index][user_index],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                     )) if numpy_generated_encoded_sequences is not None else None
                )
                logger.trace("\"{}\"->{}: {}", topic, user_id, ret[topic][user_id][-1])

        logger.success("DONE with inference ({} topics, {} users, {} arguments per user per topic)",
                       len(topics_raw), len(user_ids), generation_config.num_return_sequences)

        return ret
