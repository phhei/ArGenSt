from collections import defaultdict
from typing import Set

import numpy
from loguru import logger
from pathlib import Path

from json import load as json_load

from pprint import pformat

from pandas import read_csv


# first
# rsync -rv --max-size=50m . files.techfak.de:/media/remote/pheinisch/DebateOrgStanceArgGenerationTrier/.out/mainexperiments


def default_to_regular(d):
    if isinstance(d, defaultdict):
        return {k: default_to_regular(v) for k, v in d.items()}
    return d


def load_filled_profiles_users() -> Set[str]:
    return set(read_csv(
        filepath_or_buffer=Path("data/user_with_num_of_not_null_entries_11.csv"),
        index_col="user_id",
        encoding="utf-8",
        sep=";"
    ).index.tolist())


if __name__ == '__main__':
    main_exp_dir = Path(".out/mainexperiments")

    ret = defaultdict(lambda: defaultdict(list))
    filled_profiles_users: Set[str] = load_filled_profiles_users()

    for stats in main_exp_dir.rglob(pattern="*.json"):
        if stats.name != "stats.json":
            logger.info("Unusual file - slipping \"{}\"", stats.name)
            continue

        logger.debug("Reading {} now", stats)
        with stats.open(mode="r", encoding="utf-8") as stats_file:
            stats_dict = json_load(fp=stats_file)
        logger.trace("Stats dict: {}", stats_dict.keys())

        try:
            metrics = stats_dict["_end"]["test_inference"]["metrics"]
            logger.debug("Found {} metrics", len(metrics))

            stance = metrics["stance_F1"]
            stance_high_profiles = [v for k, v in metrics.items()
                                    if k.startswith("stance_F1_USER_") and
                                    k[len("stance_F1_USER_"):] in filled_profiles_users]
            bert = metrics["argument_bertscore_f1"]

            ret["->".join(stats.parts[2:-2])]["stance"].append(stance)
            ret["->".join(stats.parts[2:-2])]["stance_filled_profiles"].append(
                sum(stance_high_profiles) / max(1, len(stance_high_profiles))
            )
            ret["->".join(stats.parts[2:-2])]["argument"].append(bert)
        except KeyError:
            logger.opt(exception=True).warning("KeyError in {} -- skipping {}", stats.stem, stats.absolute())
            continue

    for exp, exp_stats in ret.items():
        for exp_stat_key, exp_stat_value_list in exp_stats.copy().items():
            logger.info("Experiment {} has {} {}-values", exp, len(exp_stat_value_list), exp_stat_key)
            logger.debug("Average {} value: {}", exp_stat_key, sum(exp_stat_value_list) / len(exp_stat_value_list))

            exp_stat_value_list_numpy = numpy.fromiter(iter=exp_stat_value_list, dtype=float)

            ret[exp][f"_avg_{exp_stat_key}"] = exp_stat_value_list_numpy.mean()
            ret[exp][f"_std_{exp_stat_key}"] = exp_stat_value_list_numpy.std()

    logger.success("Final result:\n{}", pformat(default_to_regular(ret), indent=4, sort_dicts=True, compact=False))
