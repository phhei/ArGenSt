from typing import Union, Iterable, Dict

import click
from pathlib import Path
from json import load as json_load

from loguru import logger

from forge import fsignature
from pipeline.Utils import convert_dict_types
from Utils import enroll_path

from pipeline.Trainer import Trainer
from pipeline.MainModule import StanceArgumentGeneratorModule
from pipeline.DataloaderProcessor import ArgumentStanceDataset

from random import seed as set_random_seed
from torch import manual_seed as set_torch_seed

force_recompution = False


@click.command()
@click.argument("configs", required=True, type=Path, nargs=-1)
@logger.catch
def start(configs: Union[Path, Iterable[Path]]) -> None:
    """
    Runs the whole procedure incl. training and inference the StanceArgumentModel.

    :param configs: provide one or more path to config-files (json-format) here.
    Each config-file is executed separately (one after other), and each config-file can execute several times by setting
    the param "runs" at the top-level. Further important top-level params should be:

    - "dataset" (including all args for initializing the dataset),
    - "call_module" (all (sub-)args for creating a StanceArgumentModule based on a recommender architecture, consisting
      "argument_generator_module" and "stance_module", including args for "SBERTTower" (text-processor),
      "SimpleFriendshipGraph" (user-id-processor), "SBERTTower" or "Linear" for processing the user properties and
      "MatMulLayer"/"SimpleNN"/"DeepCrossNetwork" for combing all the separately processed parts),
    - "metrics" for defining the metrics,
    - "training" for configure the training args (including output settings, epoch number, batch size but also
      additional settings for the .generate()-text generation).

    There are many possible settings.
    Hence, it's recommended to try various ones but processing many configs.

    :return: if configured, nice created files in your .out-folder :)
    """
    configs = enroll_path(path_input=configs)

    logger.info("Let's get started!")

    for config in configs:
        logger.info("Read the config \"{}\"", config)
        if config.exists():
            with config.open(mode="r", encoding="utf-8") as config_stream:
                dict_config: Dict = json_load(fp=config_stream)

            logger.info("Read the config: [{}]", ", ".join(map(lambda k: "\"{}\": ...".format(k), dict_config.keys())))
            random_seed = dict_config.get("random_seed", None)
            if random_seed is not None:
                try:
                    random_seed = int(random_seed)
                    set_random_seed(random_seed)
                    logger.info("Successfully sets the random seed to {} -- now you're reproducible!",
                                set_torch_seed(random_seed).seed())
                except ValueError:
                    logger.opt(exception=True).warning("Random seed must be a int-number! "
                                                       "Disable the reproducibility-feature!")

            try:
                number_runs = int(dict_config.get("runs", 1))
            except ValueError:
                logger.opt(exception=True).error("Can't parse \"runs\"=\"{}\"", dict_config.get("runs"))
                number_runs = 1

            for run in range(number_runs):
                logger.info("OK, let's do the {}/{} run with \"{}\"", run+1, number_runs, config.name)

                trainer = Trainer(
                    dataset=ArgumentStanceDataset(
                        **convert_dict_types(
                            kwargs=dict_config.get("dataset", dict_config.get("data", dict())).copy(),
                            signature=fsignature(ArgumentStanceDataset)
                        )
                    ),
                    call_module=StanceArgumentGeneratorModule(
                        args=dict_config.get("call_module", dict_config.get("model", dict())).copy()
                    ),
                    metric_args=dict_config.get("metrics", dict_config.get("metric_args", dict())).copy(),
                    training_args=dict_config.get("training", dict_config.get("training_args", dict())).copy()
                )

                if number_runs > 1:
                    logger.trace("Multiple runs, we have to prevent dir-save-conflicts")
                    new_root_dir = trainer.get_root_dir().parent.joinpath(
                        "_multirun_{}".format(trainer.get_root_dir().name), "Run-{}".format(run)
                    )

                    if not force_recompution:
                        logger.trace("Check new destination: {}", new_root_dir)
                        if new_root_dir.exists():
                            logger.warning("Destination \"{}\" already exists, skip the {}. run of {}!",
                                           new_root_dir.absolute(), run+1, config.name)
                            continue
                        logger.trace("New destination confirmed: \"{}\"", new_root_dir.name)

                    trainer.set_root_dir(
                        new_root_dir=new_root_dir
                    )
                    logger.debug("New destination confirmed: {}", trainer.get_root_dir())

                logger.info("Start all the machine!")
                trainer.train()

                logger.info("And close...")
                trainer.close()
            logger.success("Config \"{}\" done =)", config.name)
        else:
            logger.warning("\"{}\" doesn't exist, ignore!", config.absolute())


if __name__ == '__main__':
    start()
