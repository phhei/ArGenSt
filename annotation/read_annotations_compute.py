import math
from functools import reduce
from itertools import chain
from collections import Counter

from pathlib import Path
from json import load as json_load
from pandas import read_csv

from nltk.metrics.agreement import AnnotationTask
from nltk.metrics.distance import interval_distance, jaccard_distance, binary_distance
from sklearn.metrics import classification_report

from loguru import logger

path_to_annotations = Path("annotations/students-processed")
user_known_properties_path = Path("../data")
user_known_properties_file = "user_with_num_of_not_null_entries_{probs}.csv"
buckets = [(1, 3), (4, 5), (6, 9)]

if __name__ == "__main__":
    logger.info("Let's get the annotations! ({})", path_to_annotations.absolute())

    annotation_files = list(path_to_annotations.rglob(pattern="*.json"))
    logger.debug("Found {} annotators: {}", len(annotation_files), ", ".join([a.stem for a in annotation_files]))

    annotations = {
        "all": dict()
    }
    for i, _ in enumerate(buckets):
        annotations[f"bucket_{i}"] = dict()

    user_in_buckets = {
        i: set(read_csv(user_known_properties_path.joinpath(user_known_properties_file.format(probs=bucket_min)),
                        sep=";", encoding="utf-8", index_col="user_id").index) -
           (set(read_csv(user_known_properties_path.joinpath(user_known_properties_file.format(probs=bucket_max + 1)),
                         sep=";", encoding="utf-8", index_col="user_id").index)  if bucket_max < 9 else set()) -
           (set(read_csv(user_known_properties_path.joinpath(user_known_properties_file.format(probs=bucket_min - 1)),
                         sep=";", encoding="utf-8", index_col="user_id").index) -
            set(read_csv(user_known_properties_path.joinpath(user_known_properties_file.format(probs=bucket_min)),
                         sep=";", encoding="utf-8", index_col="user_id").index))
        for i, (bucket_min, bucket_max) in enumerate(buckets)
    }

    for annotation_file in annotation_files:
        logger.trace("Processing {}", annotation_file.name)
        with annotation_file.open("r", encoding="utf-8") as f:
            data = json_load(f)
        logger.success("Loaded {} annotations from \"{}\"", len(data), annotation_file.name)

        for annotation in data.values():
            sample_id = "{}->{}".format(annotation["question"], annotation["user"])
            stance_true = annotation["ground_truth_stance"]
            approach = annotation["approach_name"]
            stance_predicted = annotation["predicted_stance"]
            stance_rating = annotation["annotator_score"]["does_stance_make_sense"]["rating"]["value"]
            stance_comment = annotation["annotator_score"]["does_stance_make_sense"]["rating"]["comment"].strip("\n ")
            stance_comment = stance_comment[:300] if len(stance_comment) > 320 else stance_comment
            if stance_rating <= 2:
                argument_neg_rating = frozenset(
                    {key for key, value in annotation["annotator_score"]["does_stance_make_sense"]["no"].items()
                     if value == 1}
                )
                if annotation["annotator_score"]["does_stance_make_sense"]["no"]["why_does_stance_not_make_sense__comment"] != "comment..." and \
                        annotation["annotator_score"]["does_stance_make_sense"]["no"]["why_does_stance_not_make_sense__comment"] != "":
                    argument_neg_rating = argument_neg_rating.union(
                        {(c := annotation["annotator_score"]["does_stance_make_sense"]["no"]["why_does_stance_not_make_sense__comment"].upper().strip("\n "))[:min(300, len(c))]}
                    )
                if len(argument_neg_rating) == 0:
                    argument_neg_rating = frozenset({"no_reason_given"})
                argument_pos_rating_recall = 0
                argument_pos_rating_precision = 0
            else:
                argument_neg_rating = frozenset({"stance_fits"})
                argument_pos_rating_recall = (
                    annotation)["annotator_score"]["does_stance_make_sense"]["yes"]["generated_text_contains_all_from_original"]["rating"]
                argument_pos_rating_precision = (
                    annotation)["annotator_score"]["does_stance_make_sense"]["yes"]["generated_text_contains_realistic_additional_elements"]["rating"]

            for insert_key in ["all",
                               *[f"bucket_{i}" for i, bucket_user_set in user_in_buckets.items()
                                 if annotation["user"] in bucket_user_set]]:
                logger.trace("Inserting annotation for {} into \"{}\"", sample_id, insert_key)
                if sample_id not in annotations[insert_key]:
                    annotations[insert_key][sample_id] = {"stance_true": stance_true}
                    annotations[insert_key][sample_id][approach] = {
                        "stance_predicted": stance_predicted,
                        "stance_rating": {annotation_file.stem: stance_rating},
                        "stance_comment": {annotation_file.stem: stance_comment},
                        "argument_neg_rating": {annotation_file.stem: argument_neg_rating},
                        "argument_pos_rating_recall": {annotation_file.stem: argument_pos_rating_recall},
                        "argument_pos_rating_precision": {annotation_file.stem: argument_pos_rating_precision}
                    }
                else:
                    if approach not in annotations[insert_key][sample_id]:
                        annotations[insert_key][sample_id][approach] = {
                            "stance_predicted": stance_predicted,
                            "stance_rating": {annotation_file.stem: stance_rating},
                            "stance_comment": {annotation_file.stem: stance_comment},
                            "argument_neg_rating": {annotation_file.stem: argument_neg_rating},
                            "argument_pos_rating_recall": {annotation_file.stem: argument_pos_rating_recall},
                            "argument_pos_rating_precision": {annotation_file.stem: argument_pos_rating_precision}
                        }
                    else:
                        annotations[insert_key][sample_id][approach]["stance_rating"][annotation_file.stem] = stance_rating
                        annotations[insert_key][sample_id][approach]["stance_comment"][annotation_file.stem] = stance_comment
                        annotations[insert_key][sample_id][approach]["argument_neg_rating"][annotation_file.stem] = argument_neg_rating
                        annotations[insert_key][sample_id][approach]["argument_pos_rating_recall"][annotation_file.stem] = argument_pos_rating_recall
                        annotations[insert_key][sample_id][approach]["argument_pos_rating_precision"][annotation_file.stem] = argument_pos_rating_precision

        logger.success("Inserted {} annotations from \"{}\"", len(data), annotation_file.name)
    logger.success("Gathered all annotations")

    for bucket, annotation_data in annotations.items():
        logger.info("#"*80)
        logger.info("Processing annotation bucket \"{}\"", bucket)
        logger.info("#" * 80)
        for task, task_type in [("stance_rating", "scala"), ("stance_comment", "text"), ("argument_neg_rating", "set"),
                                ("argument_pos_rating_recall", "scala"),
                                ("argument_pos_rating_precision", "scala")]:
            logger.debug("Processing annotation task \"{}\"", task)
            approaches = list(reduce(lambda s1, s2: s1.intersection(s2),
                                     {frozenset(app.keys()) for app in annotation_data.values()},
                                     {"avoidOverlap_positiveFewShots", "avoidOverlap_zeroShots", "fineTunedApproach"}))
            logger.debug("Found {} approaches: {}", len(approaches), "/ ".join(approaches))
            for approach in approaches:
                logger.debug("Processing approach \"{}\"", approach)
                data = [[(coder, sample_item, value) for coder, value in approach_data[approach][task].items()]
                        for sample_item, approach_data in annotation_data.items()]
                data = list(chain(*data))
                logger.trace("Data annotations: {} ({}->{}->{})", len(data), bucket, task, approach)
                try:
                    logger.success("The IAA (Fleiss kappa) for bucket \"{}\", task \"{}\" and approach \"{}\" is {}",
                                   bucket, task, approach,
                                   round(AnnotationTask(
                                       data=data,
                                       distance=jaccard_distance if task_type == "set" else
                                       (interval_distance if task_type == "scala" else binary_distance)
                                   ).alpha(), 4))
                except ValueError:
                    logger.opt(exception=True).warning("The IAA (Fleiss kappa) for bucket \"{}\", task \"{}\" and "
                                                       "approach \"{}\" could not be calculated -- empty data?",
                                                       bucket, task, approach)
                except ZeroDivisionError:
                    logger.opt(exception=True).error("The IAA (Fleiss kappa) for bucket \"{}\", task \"{}\" and "
                                                     "approach \"{}\" could not be calculated: {}",
                                                     bucket, task, approach, data)
            if task == "stance_rating":
                logger.trace("Now let's calculate our performance on the {} approaches", len(approaches))
                stances_eval = {approach: ([int(d[approach]["stance_predicted"] >
                                                (.4890013039112091 if approach == "fineTunedApproach" else .5))
                                            for d in annotation_data.values()],
                                           [d["stance_true"] for d in annotation_data.values()])
                                for approach in approaches}
                for approach, (y_pred, y_true) in stances_eval.items():
                    logger.trace("Calculating hard stance performance for approach \"{}\"", approach)
                    logger.trace("y_true: {}", y_true)
                    logger.trace("y_pred: {}", y_pred)
                    logger.success("The stance classification report {}->{} for approach \"{}\" is:\n{}",
                                   bucket, task, approach,
                                   classification_report(y_true, y_pred, labels=[0, 1], target_names=["CON", "PRO"],
                                                         digits=4, output_dict=False, zero_division=0))
                stances_eval = {approach: ([0 if d[approach]["stance_predicted"] < 0.1 else
                                            (1 if d[approach]["stance_predicted"] < 0.9 else 2)
                                            for d in annotation_data.values()],
                                           [d["stance_true"]*2 for d in annotation_data.values()])
                                for approach in approaches}
                for approach, (y_pred, y_true) in stances_eval.items():
                    logger.trace("Calculating soft stance performance for approach \"{}\"", approach)
                    logger.trace("y_true: {}", y_true)
                    logger.trace("y_pred: {}", y_pred)
                    logger.success("The stance classification report {}->{} for approach \"{}\" is:\n{}",
                                   bucket, task, approach,
                                   classification_report(y_true, y_pred, labels=[0, 1, 2],
                                                         target_names=["CON", "undecided", "PRO"],
                                                         digits=4, output_dict=False, zero_division=0))

            logger.trace("Now let's calculate our performance on the {} approaches on the task \"{}\"",
                         len(approaches), task)
            data_annotated = {approach: [Counter(d[approach][task].values()) for d in annotation_data.values()]
                              for approach in approaches}
            logger.trace("Gathered {} labels", sum(map(len, data_annotated.values())))

            if task_type in ["set", "text"]:
                for approach, counter_list in data_annotated.items():
                    logger.success(
                        "The report {}->{} for approach \"{}\" is:\n{}", bucket, task, approach,
                        "\n ".join(
                            map(lambda ev: f"- {ev[0]}: {ev[1]} times selected",
                                reduce(lambda c1, c2: c1+c2, counter_list, Counter()).most_common(
                                    n=None if task_type == "set" else 10
                                )))
                    )
            elif task_type == "scala":
                for approach, counter_list in data_annotated.items():
                    logger.success(
                        "The report {}->{} for approach \"{}\" is {}:\nAVERAGE: {}\nMAJORITY VOTE: {}\nMIN: {}\n"
                        "MAX: {}",
                        bucket, task, approach,
                        round(sum(map(lambda ev: sum([k*v for k, v in ev.items()])/sum(ev.values()), counter_list))/len(counter_list), 4),
                        "\t".join(map(lambda ev: f"{round(ev[0], 2)}:{ev[1]}x",
                                      Counter([sum(co.elements())/len(list(co.elements()))
                                               for co in counter_list]).most_common(n=10))),
                        "\t".join(map(lambda ev: f"{ev[0]}:{ev[1]}x",
                                      Counter([co.most_common(n=1)[0][0]
                                               for co in counter_list]).most_common(n=None))),
                        "\t".join(map(lambda ev: f"{ev[0]}:{ev[1]}x",
                                      Counter([min(co.elements())
                                               for co in counter_list]).most_common(n=None))),
                        "\t".join(map(lambda ev: f"{ev[0]}:{ev[1]}x",
                                      Counter([max(co.elements())
                                               for co in counter_list]).most_common(n=None)))
                    )
            else:
                logger.warning("Unknown task type \"{}\" -- not stats", task_type)
        logger.info("#" * 80)
    logger.info("#" * 80)
    logger.info("#" * 80)
    logger.info("#" * 80)
