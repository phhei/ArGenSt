from typing import Dict, Any, Type, List, Iterable, Union, Tuple

import numpy
import torch
from forge import FSignature
from pathlib import Path
from loguru import logger

import json


class SetMLEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, numpy.number):
            try:
                if isinstance(obj, numpy.float32):
                    return float(obj)
                else:
                    return int(obj)
            except ValueError:
                return 0
        elif isinstance(obj, Path):
            return str(obj.absolute())
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            logger.opt(exception=True).debug("Can't json {}", obj)
            return "<<<json aborted>>>"


def convert_dict_types(kwargs: Dict, signature: FSignature) -> Dict:
    """
    Converts a (string) dict to parameters fitting a class/ method signature.

    :param kwargs: User-defined parameters (strings)
    :param signature: the class/ method signature including type-hints
    :return: a type-converted dict
    """
    ret = dict()
    logger.trace("Got {} params and {} signature fields", len(kwargs), len(signature.parameters))
    for name, param in signature.parameters.items():
        if name in kwargs:
            logger.trace("Found \"{}\" ({})", name, param)

            try:
                if isinstance(kwargs[name], param.type):
                    ret[name] = kwargs[name]
                else:
                    logger.debug("\"{}\": {} is in wrong type", name, kwargs[name])

                    ret[name] = _single_cast(value=kwargs[name], target_type=param.type)
            except TypeError:
                logger.opt(exception=True).trace("More complex type...")

                if param.type.__module__ == "typing":
                    if param.type.__class__.__name__ == "_UnionGenericAlias": # e.g. Optional
                        for i in range(len(param.type.__args__)):
                            try:
                                ret[name] = _single_cast(value=kwargs[name], target_type=param.type.__args__[i])
                                break
                            except ValueError:
                                logger.opt(exception=False).debug(
                                    "Can't handle the type \"{}\"... (you might have a real Union here)",
                                    param.type.__args__[i]
                                )
                                if i == len(param.type.__args__)-1:
                                    logger.opt(exception=True).warning("Skip \"{}\"", name)
                    elif param.type.__class__.__name__ == "_LiteralGenericAlias": # Literal
                        ret[name] = _single_cast(value=kwargs[name], target_type=str)
                    else:
                        ret[name] = _single_cast(value=kwargs[name], target_type=param.type)
                else:
                    logger.opt(exception=True).error("Can't handle this type...")
        else:
            logger.debug("{} is missing in the configs. Using the default... however, if no default is defined, "
                         "you'll get an exception!", param)

    logger.trace("Iterated over {} params, get a type-converted dict with {} values",
                 len(signature.parameters), len(ret))

    return ret


def _single_cast(value: Any, target_type: Union[Type, Tuple[Type]]):
    if isinstance(value, str) and value.lower() == "none":
        logger.trace("None-Value detected ({}), returning None while we expect the class \"{}\"",
                     value, target_type)
        return None

    if type(target_type).__module__ == "builtins" and isinstance(target_type, Tuple):
        logger.debug("We have to process {} types", len(target_type))
        if isinstance(value, List) and len(value) == len(target_type):
            return [_single_cast(value=v, target_type=target_type[i]) for i, v in enumerate(value)]
        elif len(target_type) >= 1:
            logger.info("Just consider target class {} (truncate {})", target_type[0], len(target_type)-1)
            target_type = target_type[0]
        else:
            logger.warning("No target type given! Just return the value ({})", value)
            return value

    logger.trace("Continue with a single target type {} (not multiple given anymore)", target_type)

    if target_type != str and (hasattr(target_type, "_name")
                               and target_type._name in ["List", "Tuple", "Set", "Iterable"]):
        if isinstance(value, Iterable):
            try:
                if target_type.__module__ == "typing" and len(target_type.__args__) >= 1:
                    if len(value) == len(target_type.__args__):
                        return tuple(_single_cast(value=value[i], target_type=target_type.__args__[i])
                                     for i in range(len(value)))
                    elif len(target_type._name) == "Set":
                        return {_single_cast(value=i, target_type=target_type.__args__[0]) for i in value}
                    else:
                        return [_single_cast(value=i, target_type=target_type.__args__[0]) for i in value]
                else:
                    return value
            except AttributeError:
                logger.opt(exception=False).debug("Type was not further specified")
                return value
        elif isinstance(value, str):
            return [s.strip() for s in value.split(sep=",")]
        else:
            logger.warning("Can't convert \"{}\" into a iterable ({})", value, target_type)
            return None

    if not isinstance(value, str):
        try:
            if isinstance(value, target_type):
                logger.trace("\"{}\" is already in the correct type {}", value, target_type)
                return value
        except TypeError:
            logger.opt(exception=True).warning("Casting difficulties...")
        logger.warning("Value \"{}\" is not a string -- convert...", value)
        value = str(value)
    try:
        if target_type == Any or target_type == str:
            return value
        elif target_type == bool:
            return value.lower() == "true"
        elif target_type == int or target_type == float or target_type == Path:
            return target_type(value)
        elif target_type == torch.Tensor:
            return torch.tensor(data=_single_cast(value=value, target_type=List[float]),
                                device="cuda" if torch.cuda.is_available() else "cpu")
        elif target_type == torch.nn.Module:
            if value.lower() == "identity" or value.lower() == "linear":
                return torch.nn.Identity()
            elif value.lower() == "relu":
                return torch.nn.ReLU()
            elif value.lower() == "gelu":
                return torch.nn.GELU()
            elif value.lower() == "sigmoid":
                return torch.nn.Sigmoid()
            elif value.lower() == "tanh":
                return torch.nn.Tanh()
            elif value.lower().startswith("softmax"):
                return torch.nn.Softmax(dim=int(value[7:]))
            else:
                logger.debug("Values \"{}\" doesn't match to one of the predefined values in a torch-modules", value)
                return torch.nn.Identity()
        elif str(target_type) == "typing.Callable[[torch.Tensor, int], torch.Tensor]":
            if value.lower() == "sum":
                return torch.sum
            elif value.lower() == "mean":
                return torch.mean
            elif value.lower() == "max":
                def f_max(t, i):
                    values, indices = torch.max(t, i)
                    return values
                return f_max
            elif value.lower() == "min":
                def f_min(t, i):
                    values, indices = torch.min(t, i)
                    return values
                return f_min
            else:
                logger.debug("Values \"{}\" doesn't match to one of the predefined values in a torch-function", value)
                return torch.std
        else:
            logger.warning("We don't know how to handle type \"{}\" - try and maybe fail...", target_type)
            return target_type(value)
    except TypeError:
        logger.opt(exception=True).error("Can't cast to {}", target_type)
        return value
    except IndexError:
        logger.opt(exception=True).error("Something is missing here: \"{}\"", value)
        return None
