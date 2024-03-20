import os
from numbers import Number
from pathlib import Path
from time import sleep
from typing import Dict, Optional, List, Literal, Any, Union, Tuple
from collections import defaultdict

import numpy
import torch
import pandas
from forge import fsignature
from loguru import logger
from torch.cuda import is_available as gpu_available
from tqdm import tqdm
from json import dump as json_dump
from pprint import pformat
from shutil import move

from pipeline.MainModule import StanceArgumentGeneratorModule
from pipeline.DataloaderProcessor import ArgumentStanceDataset
from pipeline.Utils import SetMLEncoder, convert_dict_types

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from rouge_score.rouge_scorer import RougeScorer
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from bert_score.scorer import BERTScorer


class UniversalStanceArgumentMetric:
    def __init__(
            self,
            calculate_topic_wise: bool = True,
            calculate_user_wise: bool = True,
            return_stance_accuracy: bool = True,
            return_stance_f1: bool = True,
            return_stance_precision: bool = True,
            return_stance_recall: bool = True,
            return_argument_rouge: bool = True,
            return_argument_blue_score: bool = False,
            return_argument_meteor: bool = False,
            return_argument_bert_score: bool = True,
            bert_score_args: Optional[Dict] = None,
            return_argument_bary_score: bool = False
    ):
        """
        Inits the universal metrics
        :param calculate_topic_wise: flag to calculate scores for every single topic, too
        :param calculate_user_wise: flag to calculate scores for every single user, too
        :param return_stance_accuracy: flag to enable accuracy calculation regarding the stances
        :param return_stance_f1: flag to enable f1 calculation regarding the stances
        :param return_stance_precision: flag to enable precision calculation regarding the stances
        :param return_stance_recall: flag to enable recall calculation regarding the stances
        :param return_argument_rouge: flag to enable the rouge-score calculation of the arguments
        --> see https://aclanthology.org/W04-1013/
        (text - if multiple generated tries are given, the min, mean and max value is calculated)
        :param return_argument_blue_score: flag to enable the blue-score calculation of the arguments
        --> see https://aclanthology.org/P02-1040/
        (text - if multiple generated tries are given, the min, mean and max value is calculated)
        :param return_argument_meteor: flag to enable the meteor-score calculation of the arguments
        --> see https://aclanthology.org/W05-0909/
        (text - if multiple generated tries are given, the min, mean and max value is calculated)
        :param return_argument_bert_score: flag to enable the BERT-score calculation of the arguments
        --> see https://openreview.net/forum?id=SkeHuCVFDr
        (text - if multiple generated tries are given, the min, mean and max value is calculated)
        :param bert_score_args: When BERTscore is enabled, you can define here additional args as dict which are
        forwarded to the BERTscore-initialization (for example, the model which should be used).
        For details, consider https://github.com/Tiiiger/bert_score (bert_score/bert_score/scorer.py)
        :param return_argument_bary_score: flag to enable the BARY-score calculation of the arguments
        --> see https://aclanthology.org/2021.emnlp-main.817/
        (text - if multiple generated tries are given, the min, mean and max value is calculated)
        """
        self.calculate_topic_wise: bool = calculate_topic_wise
        self.calculate_user_wise: bool = calculate_user_wise
        self.return_stance_accuracy: bool = return_stance_accuracy
        self.return_stance_f1: bool = return_stance_f1
        self.return_stance_precision: bool = return_stance_precision
        self.return_stance_recall: bool = return_stance_recall
        self.return_argument_rouge: bool = return_argument_rouge
        if self.return_argument_rouge:
            self.rouge: RougeScorer = RougeScorer(
                rouge_types=["rouge1", "rouge2", "rougeL"],
                use_stemmer=False,
                split_summaries=False
            )
            logger.debug("Initialized a ROUGE-Scorer ({})", self.rouge.rouge_types)
        self.return_argument_blue_score: bool = return_argument_blue_score
        self.return_argument_meteor: bool = return_argument_meteor
        self.return_argument_bert_score: bool = return_argument_bert_score
        if self.return_argument_bert_score:
            if bert_score_args is None:
                bert_score_args = {
                    "model_type": "microsoft/deberta-large-mnli",
                    "num_layers": 18,
                    "device": "cuda" if gpu_available() else "cpu",
                    "rescale_with_baseline": True,
                    "lang": "en",
                    "batch_size": 8 if gpu_available() else 32
                }
                logger.info("No Arguments for the BERT-Scorer given. Take following ones: {}", bert_score_args)
            else:
                bert_score_args = convert_dict_types(kwargs=bert_score_args, signature=fsignature(BERTScorer))
                logger.debug("Use following args for BERTScorer: {}", bert_score_args)
            self.bert_scorer = BERTScorer(**bert_score_args)
            logger.debug("Have a BERTScorer now: {}", self.bert_scorer.model_type)
        self.return_argument_bary_score: bool = return_argument_bary_score
        if self.return_argument_bary_score:
            raise NotImplementedError("See https://github.com/PierreColombo/nlg_eval_via_simi_measures")

    def __call__(
            self,
            predictions: Dict[str, Dict[Any, Tuple[Union[float, numpy.ndarray],
                                                   Optional[Union[numpy.ndarray, List[str]]]]]],
            target: Dict[str, Dict[Any, Tuple[Union[int, numpy.ndarray], Optional[str]]]],
            stance_threshold: float
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Measures the performance of the predictions considering the ground-truth targets
        :param predictions: The predictions as a dict:
        {topic: {user: (stance_prediction, argument_predictions (can be more than one, optional))}, ...}
        :param target: The ground truth as a dict:
        {topic: {user: (target_stance, target_argument (optional))},  ...}
        :param stance_threshold: a threshold which should be applied to convert the predicted stance probabilities into a stance prediction
        :return: a metric dict on the form of {metric_name: score (or a list of scores),...}
        """
        ret = dict()
        logger.trace("Start calculating scores for {} topics and {} users in total (using the stance threshold of {})",
                     len(predictions), sum(map(lambda user_dict: len(user_dict), predictions.values())),
                     str(round(stance_threshold, 4)))

        temp_stances_pred = []
        temp_stances_target = []
        overall_min_argument_scores = defaultdict(list)
        overall_mean_argument_scores = defaultdict(list)
        overall_max_argument_scores = defaultdict(list)

        if self.calculate_topic_wise:
            temp_topic = defaultdict(lambda: defaultdict(list))
            temp_topic_stances_pred = defaultdict(list)
            temp_topic_stances_target = defaultdict(list)
        if self.calculate_user_wise:
            temp_user = defaultdict(lambda: defaultdict(list))
            temp_user_stances_pred = defaultdict(list)
            temp_user_stances_target = defaultdict(list)
        for topic, topic_users in predictions.items():
            logger.trace("Scanning topic \"{}\"", topic)
            for user_id, (user_pred_stance, user_pred_arguments) in topic_users.items():
                logger.trace("\"{}\"->{}", topic, user_id)
                try:
                    ground_truth_stance, ground_truth_argument = target[topic][user_id]
                    flat_stance_pred = \
                        int(user_pred_stance > stance_threshold) if numpy.isscalar(user_pred_stance) else \
                            int(user_pred_stance[-1] > stance_threshold)
                    flat_stance_target = \
                        int(ground_truth_stance) if numpy.isscalar(ground_truth_stance) else \
                            int(ground_truth_stance[-1])

                    temp_stances_pred.append(flat_stance_pred)
                    temp_stances_target.append(flat_stance_target)
                    if self.calculate_topic_wise:
                        # noinspection PyUnboundLocalVariable
                        temp_topic_stances_pred[topic].append(flat_stance_pred)
                        # noinspection PyUnboundLocalVariable
                        temp_topic_stances_target[topic].append(flat_stance_target)
                    if self.calculate_user_wise:
                        # noinspection PyUnboundLocalVariable
                        temp_user_stances_pred[user_id].append(flat_stance_pred)
                        # noinspection PyUnboundLocalVariable
                        temp_user_stances_target[user_id].append(flat_stance_target)

                    if user_pred_arguments is None or ground_truth_argument is None: # TODO: hier
                        logger.info("Prediction or target argument is missing for user \"{}\" for topic \"{}\"",
                                    user_id, topic)
                    elif isinstance(user_pred_arguments, numpy.ndarray):
                        logger.warning("Can't handle the prediction argument format for user \"{}\" for topic \"{}\"",
                                       user_id, topic)
                    else:
                        def add_scores(add_key: str, add_score_list: List[float]):
                            logger.trace("Have to add the min, max and mean out of {} scores to key \"{}\"",
                                         len(add_score_list), add_key)
                            overall_mean_argument_scores[add_key].append(sum(add_score_list) / len(add_score_list))
                            overall_min_argument_scores[add_key].append(min(add_score_list))
                            overall_max_argument_scores[add_key].append(max(add_score_list))
                            if self.calculate_topic_wise:
                                # noinspection PyUnboundLocalVariable
                                temp_topic[topic][add_key].append(overall_mean_argument_scores[k][-1])
                            if self.calculate_user_wise:
                                # noinspection PyUnboundLocalVariable
                                temp_user[user_id][add_key].append(overall_mean_argument_scores[k][-1])
                        logger.trace("Evaluating {} predicted arguments for user \"{}\" for topic \"{}\"",
                                    len(user_pred_arguments), user_id, topic)
                        if self.return_argument_rouge:
                            rouge_scores = defaultdict(list)
                            for pred in user_pred_arguments:
                                rouge_scores.update(
                                    {k: rouge_scores[k] + [score.fmeasure] for k, score in
                                     self.rouge.score(prediction=pred, target=ground_truth_argument).items()}
                                )
                            for k, score_list in rouge_scores.items():
                                add_scores(add_key=k, add_score_list=score_list)
                        if self.return_argument_blue_score:
                            add_scores(add_key="blue", add_score_list=[sentence_bleu(
                                references=word_tokenize(ground_truth_argument),
                                hypothesis=word_tokenize(pred)
                            ) for pred in user_pred_arguments])
                        if self.return_argument_meteor:
                            add_scores(add_key="meteor", add_score_list=[meteor_score(
                                references=[word_tokenize(ground_truth_argument)],
                                hypothesis=word_tokenize(pred)
                            ) for pred in user_pred_arguments])
                        if self.return_argument_bert_score:
                            try:
                                (precision, recall, f1), bert_hash = self.bert_scorer.score(
                                    cands=user_pred_arguments,
                                    refs=[ground_truth_argument]*len(user_pred_arguments),
                                    return_hash=True
                                )
                                logger.debug("Computed the BERTscore -- {} candidates for user \"{}\" ({})",
                                            len(user_pred_arguments), user_id, bert_hash)
                                add_scores(add_key="bertscore_precision", add_score_list=precision.tolist())
                                add_scores(add_key="bertscore_recall", add_score_list=recall.tolist())
                                add_scores(add_key="bertscore_f1", add_score_list=f1.tolist())
                            except RuntimeError:
                                logger.opt(exception=True).warning("Failed to calculate the BERTscores ({}->{})",
                                                                   topic, user_id)
                except KeyError:
                    logger.opt(exception=True).warning("No target value given for \"{}\"->{}", topic, user_id)
                except ValueError:
                    logger.opt(exception=True).error("A metric is malformed...")

        logger.success("Gathered all predictions/ targets ({})", len(temp_stances_pred))
        if self.return_stance_accuracy or self.return_stance_f1 or \
                self.return_stance_precision or self.return_stance_recall:
            functions_to_check = dict()
            if self.return_stance_accuracy:
                functions_to_check["accuracy"] = accuracy_score
            if self.return_stance_f1:
                functions_to_check["F1"] = f1_score
            if self.return_stance_precision:
                functions_to_check["precision"] = precision_score
            if self.return_stance_recall:
                functions_to_check["recall"] = recall_score

            for func_name, func_score in functions_to_check.items():
                if func_name == "accuracy":
                    ret["stance_accuracy"] = accuracy_score(
                        y_true=temp_stances_target, y_pred=temp_stances_pred, normalize=True
                    )
                    logger.debug("{} stances were correctly predicted",
                                 accuracy_score(y_true=temp_stances_target, y_pred=temp_stances_pred, normalize=False))
                else:
                    ret["stance_{}".format(func_name)] = func_score(
                        y_true=temp_stances_target, y_pred=temp_stances_pred,
                        pos_label=1, average="macro"
                    )
                    ret["stance_{}_PRO".format(func_name)] = func_score(
                        y_true=temp_stances_target, y_pred=temp_stances_pred,
                        pos_label=1, average="binary"
                    )
                    ret["stance_{}_CON".format(func_name)] = func_score(
                        y_true=temp_stances_target, y_pred=temp_stances_pred,
                        pos_label=0, average="binary"
                    )
                    logger.debug("The overall macro {}-score (stances) is {}",
                                 func_name,
                                 str(round(ret["stance_{}".format(func_name)], 3)))

                if self.calculate_topic_wise and func_name != "accuracy":
                    ret_add = dict()
                    for topic in temp_topic_stances_target.keys():
                        try:
                            ret_add["stance_{}_TOPIC_{}".format(func_name, topic)] = func_score(
                                y_true=temp_topic_stances_target[topic], y_pred=temp_topic_stances_pred[topic],
                                pos_label=1, average="macro"
                            )
                        except KeyError:
                            logger.opt(exception=True).warning("Can't calculate {} for topic \"{}\"", func_name, topic)
                    logger.trace("OK, calculated {} for {} topics: {}",
                                 func_name,
                                 len(temp_topic_stances_pred),
                                 "#".join(map(lambda s: str(round(s, 2)), ret_add.values())))
                    ret.update(ret_add)
                if self.calculate_user_wise and func_name != "accuracy":
                    ret_add = dict()
                    for user in temp_user_stances_target.keys():
                        try:
                            ret_add["stance_{}_USER_{}".format(func_name, user)] = func_score(
                                y_true=temp_user_stances_target[user], y_pred=temp_user_stances_pred[user],
                                pos_label=1, average="macro"
                            )
                        except KeyError:
                            logger.opt(exception=True).info("Can't calculate {} for user \"{}\"", func_name, user)
                    logger.trace("OK, calculated {} for {} users: {}",
                                 func_name,
                                 len(temp_user_stances_pred),
                                 "#".join(map(lambda s: str(round(s, 2)), ret_add.values())))
                    ret.update(ret_add)
            logger.info("Stance score calculation is done, too")
        elif len(temp_stances_target) >= 1:
            logger.warning("Ignore {} gathered stance scores: neither return-F1, recall nor precision is set to True!",
                           len(temp_stances_target))

        def report_argument_metrics(f_min_dict: Optional[Dict[str, List[float]]],
                                    f_mean_dict: Dict[str, List[float]],
                                    f_max_dict: Optional[Dict[str, List[float]]],
                                    f_key_prefix: Optional[str] = None) -> None:
            for metric in f_mean_dict.keys():
                scores = {"WORST": f_min_dict.get(metric, None) if f_min_dict is not None else None,
                          "": f_mean_dict.get(metric, None),
                          "BEST": f_max_dict.get(metric, None) if f_max_dict is not None else None}
                for m_score_id, m_score_list in scores.items():
                    if m_score_list is None:
                        if f_key_prefix is None:
                            logger.warning("No {} scores reported for metric \"{}\"",
                                           m_score_id, metric)
                        else:
                            logger.trace("No {} scores reported for metric \"{}\"->\"{}\"",
                                         m_score_id, f_key_prefix, metric)
                    else:
                        try:
                            ret["{}argument_{}{}".format(m_score_id, metric, f_key_prefix or "")] = \
                                sum(m_score_list) / len(m_score_list)
                            logger.trace(
                                "The {} argument generated by our models fits to {}% the ground-truth argument "
                                "(w.r.t. {})",
                                m_score_id,
                                str(round(100 * ret["{}argument_{}{}".format(m_score_id, metric, f_key_prefix or "")])),
                                metric
                            )
                        except ZeroDivisionError:
                            logger.opt(exception=True).warning("Empty {} scores or metric \"{}\"", m_score_id, metric)

        logger.debug("OK, now the scores for the arguments...")
        report_argument_metrics(
            f_min_dict=overall_min_argument_scores,
            f_mean_dict=overall_mean_argument_scores,
            f_max_dict=overall_max_argument_scores
        )
        if self.calculate_topic_wise:
            # noinspection PyUnboundLocalVariable
            for topic, mean_dict in temp_topic.items():
                logger.trace("Arguments for topic \"{}\"", topic)
                report_argument_metrics(
                    f_min_dict=None,
                    f_mean_dict=mean_dict,
                    f_max_dict=None,
                    f_key_prefix="_TOPIC_{}".format(topic)
                )
        if self.calculate_user_wise:
            # noinspection PyUnboundLocalVariable
            for user, mean_dict in temp_user.items():
                logger.trace("Arguments from user \"{}\"", user)
                report_argument_metrics(
                    f_min_dict=None,
                    f_mean_dict=mean_dict,
                    f_max_dict=None,
                    f_key_prefix="_USER_{}".format(user)
                )

        logger.success("The dictionary with {} metrics is done!", len(ret))
        if self.return_stance_f1 and self.return_stance_precision and self.return_stance_recall:
            ret["stances_true"] = temp_stances_target
            ret["stances_pred"] = temp_stances_pred

        return ret


class Trainer:
    class TrainingArgs:
        def __init__(
                self,

                max_epochs: int = 8,
                early_stopping: bool = True,
                metric_keys_for_early_stopping: Optional[List[str]] = None,

                batch_size_topics: int = 4,
                batch_size_users: int = 4,
                sort_topics_by_length: bool = True,

                learning_rate: float = 1e-4,
                scalar_stance_classifier_loss: float = 1.,
                scalar_argument_generator_loss: float = 1.,
                scalar_encoder_output_shift_impact: float = 1.,
                warmup_steps: Optional[int] = None,
                cooldown_learning_rate: bool = False,
                verbose_scheduler: bool = False,

                apply_optimal_stance_thresholds_on_split: Optional[str] = None,
                generation_args_max_length: int = 50,
                generation_args_min_length: int = 8,
                generation_args_early_stopping: bool = False,
                generation_args_do_sample: bool = False,
                generation_args_num_beams: int = 8,
                generation_args_num_beam_groups: int = 4,
                generation_args_temperature: float = 1.25,
                generation_args_renormalize_logits: bool = True,
                generation_args_top_p: float = .8,
                generation_args_eta_cutoff: float = 4e-4,
                generation_args_diversity_penalty: float = 5e-2,
                generation_args_repetition_penalty: float = 1.2,
                generation_args_no_repeat_ngram_size: int = 5,
                generation_args_num_return_sequences: int = 5,
                generation_args_output_scores: bool = False,


                do_training: bool = True,
                do_inference_train: bool = False,
                do_inference_dev: bool = False,
                do_inference_test: bool = True,
                do_inference_every_epoch: bool = False,
                out_dir: Optional[Path] = None,
                writing_logs: bool = False,
                writing_stats: bool = True,
                writing_predictions: bool = True,

                clean_checkpoints_at_end: bool = True,
                store_best_model_at_end: bool = True
        ):
            # overall training-framework
            self.max_epochs: int = max_epochs
            self.early_stopping: bool = early_stopping
            self.metric_keys_for_early_stopping: Optional[List[str]] = metric_keys_for_early_stopping
            # params for data loading
            self.batch_size_topics: int = batch_size_topics
            self.batch_size_users: int = batch_size_users
            self.sort_topics_by_length: bool = sort_topics_by_length
            # training-hyperparameters
            self.learning_rate: float = learning_rate
            self.scalar_stance_classifier_loss: float = scalar_stance_classifier_loss
            self.scalar_argument_generator_loss: float = scalar_argument_generator_loss
            self.scalar_encoder_output_shift_impact: float = scalar_encoder_output_shift_impact
            self.warmup_steps: Optional[int] = warmup_steps
            self.cooldown_learning_rate: bool = cooldown_learning_rate
            self.verbose_scheduler: bool = verbose_scheduler
            # inference-hyperparameters
            self.apply_optimal_stance_thresholds_on_split: Optional[str] = apply_optimal_stance_thresholds_on_split
            self.generation_args: Optional[Dict[str, Any]] = {
                "max_length": generation_args_max_length,
                "min_length": generation_args_min_length,
                "early_stopping": generation_args_early_stopping,
                "do_sample": generation_args_do_sample,
                "num_beams": generation_args_num_beams,
                "num_beam_groups": generation_args_num_beam_groups,
                "temperature": generation_args_temperature,
                "renormalize_logits": generation_args_renormalize_logits,
                "top_p": generation_args_top_p,
                "eta_cutoff": generation_args_eta_cutoff,
                "diversity_penalty": generation_args_diversity_penalty,
                "repetition_penalty": generation_args_repetition_penalty,
                "no_repeat_ngram_size": generation_args_no_repeat_ngram_size,
                "num_return_sequences": generation_args_num_return_sequences,
                "output_scores": generation_args_output_scores
            }
            # loop-parameters
            self.do_training: bool = do_training
            self.do_inference_train: bool = do_inference_train
            self.do_inference_dev: bool = do_inference_dev
            self.do_inference_test: bool = do_inference_test
            self.do_inference_every_epoch: bool = do_inference_every_epoch
            # output options
            self.out_dir: Optional[Path] = out_dir
            self.writing_logs: bool = writing_logs
            self.writing_stats: bool = writing_stats
            self.writing_predictions: bool = writing_predictions
            # closing options
            self.clean_checkpoints_at_end: bool = clean_checkpoints_at_end,
            self.store_best_model_at_end: bool = store_best_model_at_end

    def __init__(
            self,
            dataset: ArgumentStanceDataset,
            call_module: StanceArgumentGeneratorModule,
            metric_args: Dict,
            training_args: Dict
    ):
        logger.debug("Init trainer...")

        self.call_module: StanceArgumentGeneratorModule = call_module
        self.call_module.to(device="cuda" if gpu_available() else "cpu")
        logger.info("Prediction model is loaded: {}", (scm := str(self.call_module))[:min(len(scm)-1, 30)])

        logger.debug("Received {} metric args", len(metric_args))
        self.metric = UniversalStanceArgumentMetric(
            **convert_dict_types(kwargs=metric_args, signature=fsignature(UniversalStanceArgumentMetric))
        )

        logger.debug("Receiving {} training args", len(training_args))
        self.training_args = Trainer.TrainingArgs(
            **convert_dict_types(kwargs=training_args, signature=fsignature(Trainer.TrainingArgs))
        )

        self.dataset: ArgumentStanceDataset = dataset

        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            params=self.call_module.parameters(recurse=True),
            lr=self.training_args.learning_rate
        )
        logger.info("Initialized the optimizer: {}", self.optimizer)
        if self.training_args.warmup_steps is None and not self.training_args.cooldown_learning_rate:
            self.scheduler: Union[torch.optim.lr_scheduler.ConstantLR,
                                  torch.optim.lr_scheduler.LinearLR,
                                  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
                                  torch.optim.lr_scheduler.ConstantLR] = torch.optim.lr_scheduler.ConstantLR(
                optimizer=self.optimizer,
                factor=.8,
                total_iters=10,
                verbose=self.training_args.verbose_scheduler
            )
        elif self.training_args.warmup_steps is not None and not self.training_args.cooldown_learning_rate:
            self.scheduler: Union[torch.optim.lr_scheduler.ConstantLR,
                                  torch.optim.lr_scheduler.LinearLR,
                                  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
                                  torch.optim.lr_scheduler.ConstantLR] = torch.optim.lr_scheduler.LinearLR(
                optimizer=self.optimizer,
                start_factor=.1,
                end_factor=1.,
                total_iters=self.training_args.warmup_steps,
                verbose=self.training_args.verbose_scheduler
            )
        elif self.training_args.warmup_steps is None and self.training_args.cooldown_learning_rate:
            self.scheduler: Union[torch.optim.lr_scheduler.ConstantLR,
                                  torch.optim.lr_scheduler.LinearLR,
                                  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
                                  torch.optim.lr_scheduler.ConstantLR] = \
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer=self.optimizer,
                    T_0=self.dataset.num_batches(
                        split="train",
                        batch_size_topics=self.training_args.batch_size_topics,
                        batch_size_users=self.training_args.batch_size_users,
                        sort_topics_by_length=self.training_args.sort_topics_by_length
                    ),
                    T_mult=self.dataset.num_batches(
                        split="train",
                        batch_size_topics=self.training_args.batch_size_topics,
                        batch_size_users=self.training_args.batch_size_users,
                        sort_topics_by_length=self.training_args.sort_topics_by_length
                    ),
                    eta_min=self.training_args.learning_rate * .05,
                    verbose=self.training_args.verbose_scheduler
                )
        else:
            self.scheduler: Union[torch.optim.lr_scheduler.ConstantLR,
                                  torch.optim.lr_scheduler.LinearLR,
                                  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
                                  torch.optim.lr_scheduler.ConstantLR] = torch.optim.lr_scheduler.CyclicLR(
                optimizer=self.optimizer,
                base_lr=self.training_args.learning_rate*.1,
                max_lr=self.training_args.learning_rate,
                step_size_up=self.training_args.warmup_steps,
                step_size_down=self.dataset.num_batches(
                    split="train",
                    batch_size_topics=self.training_args.batch_size_topics,
                    batch_size_users=self.training_args.batch_size_users,
                    sort_topics_by_length=self.training_args.sort_topics_by_length
                )-self.training_args.warmup_steps,
                mode="exp_range",
                cycle_momentum=False,
                verbose=self.training_args.verbose_scheduler
            )
        logger.debug("Initialed the learning rate scheduler: {} "
                     "(see https://towardsdatascience.com/"
                     "a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863 for details)", self.scheduler)

        self.get_root_dir().mkdir(parents=True, exist_ok=True)
        if self.training_args.writing_logs:
            log_file = self.get_root_dir().joinpath("logs.txt")
            self.log_number = logger.add(
                sink=log_file,
                level="INFO",
                colorize=False,
                catch=True,
                encoding="utf-8",
                errors="replace"
            )
            logger.debug("Started logging into {}, too", log_file)
        else:
            self.log_number = None

    def train(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Trains the underlying StanceArgumentGeneratorModule by using the data, optimizer and training args.
        If configured, writes data/ weights/ logs ect. into the configured output-dir.
        :return: All the gathered results from each single epoch incl. inference The dict is structured as follows:
        Epoc/_end: (Split: (Results: Values))
        """
        logger.info("OK, let's start training!")
        ret = dict()

        optimal_threshold = None

        if self.training_args.do_training:
            last_score = -1
            for epoch_number in range(1, self.training_args.max_epochs+1):
                logger.info("Start with the {}. training epoch", epoch_number)
                ret["Epoch_{}".format(epoch_number)] = dict()
                ret["Epoch_{}".format(epoch_number)]["training"] = self.one_iteration(
                    number=epoch_number,
                    on_split="train",
                    training=True,
                    optimal_stance_threshold=optimal_threshold
                )
                logger.success("Training of {}. epoch done!", epoch_number)
                if self.training_args.do_inference_every_epoch:
                    logger.info("OK, now the inference")
                    test_dict, optimal_threshold = self.test(
                        epoch_number=epoch_number, optimal_threshold=optimal_threshold, return_optimal_threshold=True
                    )
                    ret["Epoch_{}".format(epoch_number)].update(test_dict)
                if self.training_args.early_stopping:
                    if "dev_inference" in ret["Epoch_{}".format(epoch_number)]:
                        metrics = ret["Epoch_{}".format(epoch_number)]["dev_inference"].get("metrics", dict())
                    else:
                        logger.warning("You want to do early stopping but having no stats about the dev-split! "
                                       "We have to calculate some...")
                        with torch.no_grad():
                            metrics = self.one_iteration(
                                number=epoch_number,
                                on_split="dev",
                                training=False,
                                optimal_stance_threshold=optimal_threshold
                            ).get("metrics", dict())
                    metrics_available = {k for k, v in metrics.items() if isinstance(v, float)}
                    if self.training_args.metric_keys_for_early_stopping is None:
                        used_metrics = metrics_available
                        missing_metrics = set()
                    else:
                        used_metrics = metrics_available.intersection(self.training_args.metric_keys_for_early_stopping)
                        missing_metrics = \
                            set(self.training_args.metric_keys_for_early_stopping).difference(metrics_available)
                    logger.debug("For early stopping, we consider the following {} metrics: {}",
                                 len(used_metrics), " + ".join(used_metrics))
                    if len(missing_metrics) >= 1:
                        logger.warning("{} metrics are missing (i.e. not returned by the metric-module but expected "
                                       "for calculation the early stopping performance "
                                       "OR not in a float-number format): {}",
                                       len(missing_metrics), " as well as ".join(missing_metrics))
                    if len(used_metrics) == 0:
                        logger.error("No metrics usable for score computation to consider the best model :\\ "
                                     "-- early stopping disabled for the epoch {}", epoch_number)
                    else:
                        used_metric_scores = [metrics[m] for m in used_metrics]
                        logger.trace("Fetch metric scores: {}",
                                     " + ".join(map(lambda ums: str(round(ums, 2)), used_metric_scores)))
                        new_score = sum(used_metric_scores)/len(used_metric_scores)
                        ret["Epoch_{}".format(epoch_number)]["early_stopping"] = {
                            "used_metrics": used_metrics,
                            "used_metric_scores": used_metric_scores,
                            "avg": new_score,
                            "missed_metrics": missing_metrics
                        }
                        logger.debug("Your performance: {}->{}", str(round(last_score, 4)), str(round(new_score, 4)))
                        if new_score < last_score:
                            logger.warning("Canceling the training at epoch {}. The DEV-performance is worse now ({})",
                                           epoch_number, str(round(new_score, 3)))
                            ret["Epoch_{}".format(epoch_number)]["early_stopping"]["abort_training"] = True
                            stored_model_path = ret["Epoch_{}".format(epoch_number-1)].get("early_stopping", dict()).get(
                                "stored_model_path", None)
                            if stored_model_path is None:
                                logger.warning("Epoch {}: try to restore the {}. epoch (generalized better), "
                                               "but no weights were stored!", epoch_number, epoch_number-1)
                            else:
                                missing_keys, unexpected_keys = self.call_module.load_state_dict(
                                    state_dict=torch.load(
                                        f=stored_model_path,
                                        map_location="cuda" if gpu_available() else "cpu"
                                    ),
                                    strict=False
                                )
                                logger.info("Reloaded the model weights from last epoch.")
                                if len(missing_keys) >= 1:
                                    logger.warning("{} keys were missing in the loaded state dict "
                                                   "(while restoring the model weights): {}",
                                                   len(missing_keys), " and ".join(missing_keys))
                                if len(unexpected_keys) >= 1:
                                    logger.warning("{} keys were unexpected in the loaded state dict "
                                                   "(while restoring the model weights): {}",
                                                   len(missing_keys), " and ".join(missing_keys))

                                logger.trace("We have to restore the optional threshold as well (current: {})",
                                             optimal_threshold)
                                optimal_threshold = \
                                    None \
                                        if optimal_threshold is None or \
                                           self.training_args.apply_optimal_stance_thresholds_on_split is None \
                                        else ret["Epoch_{}".format(epoch_number - 1)].get(
                                            "{}_inference".format(self.training_args.apply_optimal_stance_thresholds_on_split),
                                            ret["Epoch_{}".format(epoch_number - 1)]["training"]
                                        ).get("stance_threshold", None)
                                if optimal_threshold is None:
                                    logger.warning("Can't restore the optimal stance threshold. Is is expected when "
                                                   "\"self.training_args.apply_optimal_stance_thresholds_on_split\""
                                                   "is None")
                            break
                        else:
                            last_score = new_score
                            model_file = self.get_root_dir().joinpath("model-epoch-{}.pt".format(epoch_number))
                            try:
                                torch.save(
                                    obj=self.call_module.state_dict(),
                                    f=model_file
                                )
                                logger.info("Saved the weights into \"{}\"", model_file.name)
                                ret["Epoch_{}".format(epoch_number)]["early_stopping"]["stored_model_path"] = \
                                    str(model_file.absolute())
                            except IOError:
                                logger.opt(exception=True).error("Can't save the model! ({})",
                                                                 ", ".join(self.call_module.state_dict().keys()))
                            ret["Epoch_{}".format(epoch_number)]["early_stopping"]["abort_training"] = False
            logger.success("Training DONE!")
        else:
            logger.warning("Training is skipped ({} epochs)", self.training_args.max_epochs)

        logger.info("Let's do the inference now :)")
        if optimal_threshold is None:
            logger.warning("No optimal threshold is found so far (maybe you disabled the training?) Try .5")
        ret["_end"] = self.test(epoch_number=None, optimal_threshold=optimal_threshold, return_optimal_threshold=False)

        if self.training_args.writing_stats:
            logger.debug("OK, now let's write the stats into {}", self.get_root_dir().absolute())
            stats_path = self.get_root_dir().joinpath("stats.json")
            try:
                with stats_path.open(mode="w", encoding="utf-8") as stats_file:
                    json_dump(obj=ret, fp=stats_file, indent=2, skipkeys=True, sort_keys=True, cls=SetMLEncoder)
                logger.info("Wrote the stats into \"{}\"", stats_path)
            except IOError:
                logger.opt(exception=True).warning(
                    "Failed to write the stats: {}",
                    pformat(object=ret, indent=2, depth=3, compact=False, sort_dicts=True)
                )

        if self.training_args.writing_predictions:
            if self.training_args.do_inference_test:
                try:
                    predictions = ret["_end"]["test_inference"]["predictions"]
                    ground_truth = ret["_end"]["test_inference"]["ground_truth"]
                    prediction_folder = self.get_root_dir().joinpath("test-predictions")
                    prediction_folder.mkdir(parents=False, exist_ok=True)
                    for topic, topic_predictions in predictions.items():
                        logger.trace("\"{}\" -> {}", topic, prediction_folder)
                        try:
                            prediction_file = prediction_folder.joinpath(
                                "{}.csv".format(
                                    str(topic).replace("?", "").replace("\"", "").replace("*", "_").replace(":", "_").
                                    replace("/", "-").replace("\\", "")
                                )
                            )
                            pandas.DataFrame.from_dict(
                                data={user: [pred[0] if isinstance(pred[0], Number) else pred[0][-1]] +
                                            [ground_truth[topic][user][0]
                                             if isinstance(ground_truth[topic][user][0], Number) else
                                             ground_truth[topic][user][0][-1]] +
                                            [ground_truth[topic][user][1]] + pred[1]
                                      for user, pred in topic_predictions.items() if pred[1] is not None},
                                orient="index",
                                columns=["Pred_Stance_PRO", "Target_Stance_PRO", "Target_Argument"] +
                                        ["Pred_Arg_{}".format(i)
                                         for i in range(
                                            self.training_args.generation_args.get("num_return_sequences", 1))]
                            ).to_csv(
                                path_or_buf=prediction_file,
                                header=True,
                                index=True,
                                index_label="User",
                                encoding="utf-8"
                            )
                            logger.debug("Successfully wrote {}", prediction_file.name)
                        except IOError:
                            logger.opt(exception=True).warning(
                                "Failed to write the predictions: {}",
                                pformat(object=topic_predictions, indent=2, compact=False, sort_dicts=True)
                            )
                except KeyError:
                    logger.opt(exception=True).error("No predictions")
            else:
                logger.warning("Please set \"do_inference_test\"=True when setting \"writing_predictions\"=True")

        logger.success("Finished training process, gathered a dictionary with {} super-keys", len(ret))
        return ret

    def get_root_dir(self) -> Path:
        if self.training_args.out_dir is not None:
            return self.training_args.out_dir

        ret_path = Path(
            ".out",
            "{}+{}".format(
                self.call_module.argument_module.name_or_path,
                "stance({}-{}-{})".format(
                    len(self.call_module.stance_module.towers),
                    len(self.call_module.stance_module.combiners),
                    self.call_module.stance_module.classification_head.classification_size
                )
            ),
            "{}inference-{}".format(
                "trained-" if self.training_args.do_training else "",
                "".join(["train" if self.training_args.do_inference_train else ""] +
                        ["dev" if self.training_args.do_inference_dev else ""] +
                        ["test" if self.training_args.do_inference_test else ""])
            )
        )

        if self.training_args.do_inference_train or self.training_args.do_inference_dev \
                or self.training_args.do_inference_test:
            generation_args_string = "gen+{}".format("+".join(map(lambda kv: "-".join(map(lambda k_v: str(k_v), kv)),
                                                                  self.training_args.generation_args.items())))
            try:
                max_folder_name_length = os.pathconf("/", "PC_NAME_MAX")
            except AttributeError or ValueError:
                logger.opt(exception=True).info("Maybe you're using a Windows-OP? "
                                                "We assume a max_folder_name_length of 255")
                max_folder_name_length = 255
            if len(generation_args_string) > max_folder_name_length:
                logger.warning("The dir-name introducing the training args exceeds the max_folder_name_length of {}. "
                               "We discard \"{}\"",
                               max_folder_name_length, generation_args_string[max_folder_name_length:])
                generation_args_string = generation_args_string[:max_folder_name_length]
            return ret_path.joinpath(
                generation_args_string
            )

        return ret_path

    def set_root_dir(self, new_root_dir: Path):
        old_root_dir = self.get_root_dir()

        while new_root_dir.exists() and len(list(new_root_dir.iterdir())) >= 1:
            logger.warning("{} already exists and is not empty -- before we mess up here, let's modify the Path "
                           "\"{}\" slightly", new_root_dir, new_root_dir.name)
            new_root_dir = new_root_dir.parent.joinpath("_{}".format(new_root_dir.name))
            logger.debug("New root dir: {}", new_root_dir.name)

        if self.log_number is not None:
            logger.remove(handler_id=self.log_number)

        if old_root_dir.exists():
            logger.info("We have to move all existing files from \"{}\"->\"{}\"", old_root_dir.name, new_root_dir.name)
            move(src=old_root_dir, dst=new_root_dir)

        self.training_args.out_dir = new_root_dir

        if self.log_number is not None:
            log_file = self.get_root_dir().joinpath("logs.txt")
            self.log_number = logger.add(
                sink=log_file, level="INFO",
                colorize=False,
                catch=True,
                encoding="utf-8",
                errors="replace"
            )
            logger.debug("Redirected logging into {}, too", log_file)

        logger.success("Move from \"{}\" to \"{}\" succeeded", old_root_dir, new_root_dir)

    def test(
            self,
            epoch_number: Optional[int] = None,
            optimal_threshold: Optional[float] = None,
            return_optimal_threshold: bool = False
    ) -> Union[Dict, Tuple[Dict, float]]:
        ret = dict()

        splits_for_inference = []
        if self.training_args.do_inference_train:
            splits_for_inference.append("train")
        if self.training_args.do_inference_dev:
            splits_for_inference.append("dev")
        if self.training_args.do_inference_test:
            splits_for_inference.append("test")

        logger.info("OK, now the inference")
        if len(splits_for_inference) == 0:
            logger.warning("You want to do inference, but none of the splits are defined to be used for "
                           "inference ({}/{}/{})",
                           self.training_args.do_inference_train, self.training_args.do_inference_dev,
                           self.training_args.do_inference_test)
        for split in splits_for_inference:
            with torch.no_grad():
                ret["{}_inference".format(split)] = self.one_iteration(
                    number=epoch_number or self.training_args.max_epochs+1,
                    on_split=split,
                    training=False,
                    optimal_stance_threshold=optimal_threshold
                )
                if epoch_number is not None and split == self.training_args.apply_optimal_stance_thresholds_on_split:
                    new_optimal_threshold = ret["{}_inference".format(split)].get("stance_threshold", None)
                    logger.info("Updated optimal threshold for all further iterations: {}->{}",
                                "n/a" if optimal_threshold is None else
                                str(round(optimal_threshold, 3)),
                                "ERROR" if new_optimal_threshold is None else str(round(new_optimal_threshold, 3)))
                    optimal_threshold = new_optimal_threshold
        logger.debug("All inference splits ({}) done...", len(splits_for_inference))

        return (ret, optimal_threshold) if return_optimal_threshold else ret

    def one_iteration(self, number: int, on_split: Literal["train", "dev", "test"], training: bool,
                      optimal_stance_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Performs a pass through all database-instances of a certain split
        :param number: The number of the epoch. Should start with 1
        (a value between "1" and self.training_args.max_epochs)
        :param on_split: the selected split. Should be either "train", "dev" or "test"
        :param training: Should the split used to train the model, i.e., model weight adaptation? WARNING: needs more
        computational resources than inference (setting this flag to False)
        :param optimal_stance_threshold: the optimal stance threshold (when to treat a probability as PRO?).
        Should be a value between "0" and "1". If not given, .5 is applied
        :return: a comprehensive dictionary collecting all stats and predictions/ labels of the iteration.
        """
        logger.debug("Start with the {}. epoch", number)
        if torch.cuda.is_available():
            logger.success("Using GPUs ({} backpropagation): ", "with" if training else "w/o")
            logger.debug(torch.cuda.memory_summary())

        if training:
            self.call_module.train()
        else:
            self.call_module.eval()

        ret = {
            "epoch": number,
            "split": on_split,
            "num_instances": self.dataset.num_instances(split=on_split),
            "num_batches": self.dataset.num_batches(
                split=on_split,
                batch_size_topics=self.training_args.batch_size_topics,
                batch_size_users=self.training_args.batch_size_users,
                sort_topics_by_length=self.training_args.sort_topics_by_length
            ),
            "training": training,
            "optimizer": self.optimizer,
            "loss_history": [],
            "lr_history": [],
            "predictions": defaultdict(dict),
            "ground_truth": defaultdict(dict)
        }

        logger.trace(ret)

        try:
            for step, (topics, user_ids, user_properties, ground_truth_stances_per_topic_per_user,
                       ground_truth_arguments_per_topic_per_user) in tqdm(
                iterable=enumerate(
                    self.dataset.get_iterable(
                        split=on_split,
                        batch_size_topics=self.training_args.batch_size_topics,
                        batch_size_users=self.training_args.batch_size_users,
                        sort_topics_by_length=self.training_args.sort_topics_by_length,
                        tensors_device="cuda" if gpu_available() else "cpu"
                    )
                ),
                total=ret["num_batches"],
                unit="batch"
            ):
                logger.trace("Let's process the {}->{}. batch (topics: {}/ users: {})",
                             number, step,
                             ", ".join(map(lambda t: "\"{}\"".format(t), topics)),
                             "/".join(map(lambda u: "*{}*".format(u), user_ids)))
                if training:
                    logger.trace("Training mode")
                    self.optimizer.zero_grad()
                    try:
                        stance_probabilities, (generator_output, generator_loss) = self.call_module(
                            topics=topics,
                            user_ids=user_ids,
                            user_properties=user_properties,
                            arguments_per_topic_per_user=ground_truth_arguments_per_topic_per_user,
                            encoder_output_shift_impact=self.training_args.scalar_encoder_output_shift_impact *
                                                        (min(.33, .33*number/self.training_args.max_epochs) +
                                                         min(.66, .66*(step+1)/ret["num_batches"]))
                        )
                    except torch.cuda.OutOfMemoryError:
                        logger.opt(exception=True).critical("Ups... we need a larger GPU for this amount!")
                        logger.info(torch.cuda.memory_summary())
                        logger.warning("We skip the following users {} (with properties {}) in {} topics (target: {})",
                                       user_ids, user_properties, len(topics),
                                       ground_truth_arguments_per_topic_per_user)
                        del stance_probabilities
                        del generator_output
                        del generator_loss
                        sleep(15)
                        continue
                    logger.debug("Compute the final loss (Generator: {})", generator_loss.cpu())
                    stance_loss = torch.nn.functional.binary_cross_entropy(
                        input=stance_probabilities,
                        target=ground_truth_stances_per_topic_per_user.to(stance_probabilities.dtype)
                    ) if self.call_module.stance_module.classification_head.classification_size == 1 else \
                        torch.nn.functional.cross_entropy(
                            input=torch.flatten(input=stance_probabilities, start_dim=0, end_dim=-2),
                            target=torch.flatten(ground_truth_stances_per_topic_per_user.to(stance_probabilities.dtype))
                        )
                    logger.trace("Computed the stance-loss: {}", stance_loss)

                    loss = self.training_args.scalar_stance_classifier_loss * stance_loss + \
                           self.training_args.scalar_argument_generator_loss * generator_loss
                    logger.trace("Final loss {} = {}*{} + {}*{}",
                                 loss, self.training_args.scalar_stance_classifier_loss, stance_loss.cpu(),
                                 self.training_args.scalar_argument_generator_loss, generator_loss.cpu())
                    loss.backward()

                    self.optimizer.step()
                    logger.trace(
                        "Performed a parameter adaption with the {}. batch with a learning rate of {}",
                        step,
                        str(round(self.scheduler.get_last_lr()
                                  if isinstance(self.scheduler.get_last_lr(), float) else self.scheduler.get_last_lr()[-1], 4))
                    )
                    ret["lr_history"].append(
                        self.scheduler.get_last_lr() if isinstance(self.scheduler.get_last_lr(), float) else self.scheduler.get_last_lr()[-1]
                    )
                    self.scheduler.step()
                    if isinstance(self.scheduler.get_last_lr(), List) and len(self.scheduler.get_last_lr()) >= 2:
                        logger.trace("Adapt learning rate: {}->{}",
                                     self.scheduler.get_last_lr()[-2], self.scheduler.get_last_lr()[-1])

                    ret["loss_history"].append(loss.cpu().item())

                    logger.trace("Store the predictions")
                    stance_probabilities = stance_probabilities.cpu().detach().numpy()
                    _, max_generator_output_indices = torch.max(input=generator_output.cpu().detach(), dim=-1)
                    for i_topic, topic in enumerate(topics):
                        logger.trace("Collect predictions for topic: \"{}\"", topic)
                        for i_user, user in enumerate(user_ids):
                            ret["predictions"][topic][user] = (
                                stance_probabilities[i_topic, i_user],
                                [self.call_module.argument_tokenizer.decode(
                                    token_ids=max_generator_output_indices[i_topic, i_user],
                                    skip_special_tokens=True, clean_up_tokenization_spaces=True
                                )] if self.call_module.argument_tokenizer is not None else None
                            )
                            logger.trace("Fetched following prediction for user {}: {}",
                                         user, ret["predictions"][topic][user])
                else:
                    logger.trace("Inference-mode")
                    inference_ret =\
                        self.call_module.inference(
                            topics=topics,
                            user_ids=user_ids,
                            user_properties=user_properties,
                            encoder_output_shift_impact=self.training_args.scalar_encoder_output_shift_impact *
                                                        (min(.33, .33*number/self.training_args.max_epochs)+.66),
                            args_for_generate_function=self.training_args.generation_args,
                            return_config_for_generate_function=False
                        )

                    for topic, users in inference_ret.items():
                        for user, values in users.items():
                            ret["predictions"][topic][user] = values
                ground_truth_stances_per_topic_per_user = ground_truth_stances_per_topic_per_user.cpu().numpy()
                if isinstance(ground_truth_arguments_per_topic_per_user, torch.Tensor):
                    ground_truth_arguments_per_topic_per_user = ground_truth_arguments_per_topic_per_user.cpu().numpy()
                for i_topic, topic in enumerate(topics):
                    logger.trace("Collect labels for topic: \"{}\"", topic)
                    for i_user, user in enumerate(user_ids):
                        gta = ground_truth_arguments_per_topic_per_user[i_topic][i_user]
                        ret["ground_truth"][topic][user] = (
                            ground_truth_stances_per_topic_per_user[i_topic, i_user],
                            (None if self.call_module.argument_tokenizer is None else
                             self.call_module.argument_tokenizer.decode(token_ids=gta))
                            if isinstance(gta, torch.Tensor) else gta
                        )
                        logger.trace("Fetched following label for user {}: {}",
                                     user, ret["ground_truth"][topic][user])
                logger.trace("Finished processing the {}->{}. batch (having {} losses and {} predictions now)",
                             number, step, len(ret["loss_history"]), len(ret["predictions"]))

            logger.info("Finished the computation of the {}. epoch ({}), having predictions for {} topics now"
                        "({} total)",
                        number, on_split, len(ret["predictions"]), sum(map(len, ret["predictions"].values())))
        except RuntimeError:
            logger.opt(exception=True).error("Found a bug!")
            exit(-10)

        logger.trace("now let's compute the metrics...")

        if number <= self.training_args.max_epochs and \
                self.training_args.apply_optimal_stance_thresholds_on_split == on_split:
            old_optimal_threshold = optimal_stance_threshold

            def compute_optimal_threshold(predicted: numpy.ndarray, reference: numpy.ndarray) -> float:
                logger.trace("OK, having {} entries...", len(predicted))
                try:
                    fpr, tpr, thresholds = roc_curve(y_score=predicted, y_true=reference,
                                                     pos_label=1)
                    true_false_rate = tpr - fpr
                    ix = numpy.argmax(true_false_rate)
                    logger.trace("Found following true-positive-rates: {}, false-positive-rates: {} "
                                 "under following thresholds: {}", tpr, fpr, thresholds)
                    threshold = thresholds[ix]
                    logger.info("Found the optimal threshold: {}", round(threshold, 5))
                except ValueError:
                    logger.opt(exception=True).critical("Something went wrong in calculating the optimal threshold "
                                                        "(fall back to .5). "
                                                        "The values for predicted_arg_kp_matches: {}. "
                                                        "The values for ground_truth_arg_kp_matches: {}",
                                                        predicted.tolist(),
                                                        reference.tolist())
                    threshold = .5
                return threshold

            stances_pred = []
            stances_truth = []
            for topic, user_dict in ret["predictions"].items():
                for user, (stance, _) in user_dict.items():
                    truth, _ = ret["ground_truth"].get(topic, dict()).get(user, (None, None))
                    if truth is not None:
                        stances_pred.append(stance[-1] if isinstance(stance, numpy.ndarray) else stance)
                        stances_truth.append(truth[-1] if isinstance(truth, numpy.ndarray) else truth)

            optimal_stance_threshold = compute_optimal_threshold(
                predicted=numpy.array(stances_pred, dtype=float),
                reference=numpy.array(stances_truth, dtype=float)
            )
            logger.info("Found a new optimal threshold: {} -> {}", old_optimal_threshold, optimal_stance_threshold)
        elif optimal_stance_threshold is None:
            logger.warning("We don't have any cue about a good threshold for treating a stance probability as PRO "
                           "({} != {}). Take .5", self.training_args.apply_optimal_stance_thresholds_on_split, on_split)
            optimal_stance_threshold = .5
        else:
            logger.debug("Copy optimal stance threshold: {}", optimal_stance_threshold)

        ret["stance_threshold"] = optimal_stance_threshold
        ret["metrics"] = self.metric(
            predictions=ret["predictions"],
            target=ret["ground_truth"],
            stance_threshold=optimal_stance_threshold
        )

        if len(ret["loss_history"]) >= 2:
            ret.update({
                "min_loss": min(ret["loss_history"]),
                "max_loss": max(ret["loss_history"]),
                "mean_loss": sum(ret["loss_history"])/len(ret["loss_history"])
            })
        logger.success("Successfully finished this epoch ({}->{}). We {} the model{}",
                       number, on_split, "trained" if training else "evaluated",
                       " (with a loss of {})".format(ret.get("mean_loss", ret["loss_history"])) if training else "")

        torch.cuda.empty_cache()

        return ret

    def close(self) -> None:
        """
        Clean up the experiment. If the generated folder to the experiment is empty, it is deleted.
        If an experiment logger was set, it is removed
        :return: a clear mind :)
        """
        if self.training_args.clean_checkpoints_at_end:
            logger.debug("OK, we clean the checkpoints... are there any?")
            for number in range(1, self.training_args.max_epochs+1):
                model_file = self.get_root_dir().joinpath("model-epoch-{}.pt".format(number))
                if model_file.exists():
                    logger.info("Remove the checkpoint \"{}\"", model_file.name)
                    model_file.unlink(missing_ok=False)

        if self.training_args.store_best_model_at_end:
            logger.debug("OK, now we want to save the best model.")
            torch.save(obj=self.call_module.state_dict(), f=self.get_root_dir().joinpath("model.pt"))

        if self.log_number is not None:
            logger.remove(handler_id=self.log_number)

        if self.get_root_dir().exists() and not any(self.get_root_dir().iterdir()):
            logger.warning("Your root directory is empty - remove \"{}\"", self.get_root_dir().absolute())
            self.get_root_dir().rmdir()

        del self.call_module
