from pathlib import Path
from typing import Union, Iterable

from loguru import logger


def enroll_path(path_input: Union[Path, Iterable[Path]]) -> Iterable[Path]:
    if isinstance(path_input, Path):
        if "*" in path_input.stem:
            logger.info("A pattern as config path is given: \"{}\"", path_input.name)
            logger.debug("Working directory: {}", path_input.parent.absolute())
            configs = list(path_input.parent.glob(pattern=path_input.name))
            logger.success("Found {} configs: {}", len(configs), "/".join(map(lambda c: c.name, configs)))
        else:
            logger.info("Only a single config given: \"{}\"", path_input.name)
            configs = [path_input]
    else:
        logger.debug("{} configs given: {}", len(path_input), "/".join(map(lambda c: c.name, path_input)))
        configs = path_input

    return configs
