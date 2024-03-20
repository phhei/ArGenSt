from itertools import chain
from typing import Union, Iterable, Dict, List

import click
from pathlib import Path
from json import load as json_load

from loguru import logger

from forge import fsignature

from pipeline.DataloaderProcessor import transform_to_dataloader
from pipeline.Utils import convert_dict_types
from Utils import enroll_path

from pipeline.PromptChatGPT.process.Data import load_instances
from pipeline.PromptChatGPT.process.core import process
from pipeline.data_gathering.opinion_arguments.ExternOpinionArgument import ExternOpinionArgument
from pipeline.Trainer import UniversalStanceArgumentMetric


@click.command()
@click.argument("configs", required=True, type=Path, nargs=-1)
@logger.catch
def start(configs: Union[Path, Iterable[Path]]) -> None:
    """
    Runs the whole procedure prompting ChatGPT with stuff configured in the config.
    :param configs: config file. The main keys to configure:
    - "opinion_loading" (all args for initializing the dataset),
    - "instance_loading": all args for loading the instances to be prompted (including the path to the prompt csv)
    - "chatGPT": all args for prompting ChatGPT (including the name of the model to be used) -
    you have to set the subkey "api_key" to your OpenAI API key here!
    :return: happy automatic prompting experience :)
    """
    configs = enroll_path(path_input=configs)

    logger.info("Let's get started!")

    for config in configs:
        logger.info("Read the config \"{}\"", config)
        if config.exists():
            with config.open(mode="r", encoding="utf-8") as config_stream:
                dict_config: Dict = json_load(fp=config_stream)

            logger.info("Read the config: [{}]", ", ".join(map(lambda k: "\"{}\": ...".format(k), dict_config.keys())))

            # Data loading
            logger.trace("OK, we have to load the data...")
            data = transform_to_dataloader(
                **convert_dict_types(kwargs=dict_config.get("opinion_loading", dict()),
                                     signature=fsignature(transform_to_dataloader))
            )
            logger.trace("OK, we have the basic data!")
            grouped_opinions: Dict[str, List[ExternOpinionArgument]] = data[3]
            logger.debug("OK, we have the {} grouped opinions!", len(grouped_opinions))
            logger.trace(" # ".join(grouped_opinions.keys()))
            listed_opinions = list(chain(*grouped_opinions.values()))
            logger.debug("OK, we have the {} opinions!", len(listed_opinions))

            instances = load_instances(
                opinions=listed_opinions,
                **convert_dict_types(kwargs=dict_config.get("instance_loading", dict_config.get("loading", dict())),
                                     signature=fsignature(load_instances))
            )
            logger.info("Fetched {} instances", len(instances))

            # processing
            if "metrics" in dict_config:
                logger.info("Configure the metric with {} params", len(dict_config["metrics"]))
                metric = UniversalStanceArgumentMetric(
                    **convert_dict_types(kwargs=dict_config["metrics"],
                                         signature=fsignature(UniversalStanceArgumentMetric))
                )
            else:
                metric = None
            process_kwargs = dict_config.get("chatGPT", dict())
            for del_key in ["instances", "users", "metric_callback"]:
                if del_key in process_kwargs:
                    logger.warning("The key \"{}\" is not allowed in the config \"chatGPT\"! "
                                   "We remove {} since we set this automatically for you it this point!",
                                   del_key, process_kwargs.pop("instances"))

            process(
                instances=instances,
                users={user.url: user for user in data[2]},
                metric_callback=metric,
                **convert_dict_types(kwargs=process_kwargs,
                                     signature=fsignature(process))
            )
            logger.success("Finished the config \"{}\"!", config.name)
        else:
            logger.error("The config \"{}\" does not exist!", config)


if __name__ == '__main__':
    start()
