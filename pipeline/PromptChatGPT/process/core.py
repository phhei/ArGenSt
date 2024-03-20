import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union, Set
from json import dump as json_dump, dumps as json_dumps
from loguru import logger

import openai
from tqdm import tqdm

from pipeline.PromptChatGPT.process.Data import PromptInstance
from pipeline.data_gathering.users.UserSelfReport import UserSelfReport
from pipeline.Trainer import UniversalStanceArgumentMetric

from sklearn.metrics import classification_report


@logger.catch
def process(
        api_key: str,
        task_prompt: str,
        instances: List[PromptInstance],
        users: Dict[str, UserSelfReport],
        model: str = "gpt-3.5-turbo",
        required_user_attributes: Optional[Union[int, Set[str]]] = None,
        pos_shots: Union[int, bool] = True,
        neg_shots: Union[int, bool] = False,
        text_target: str = "conclusion",
        arguments_per_instance: int = 1,
        additional_chat_settings: Optional[Dict] = None,
        save_to_file: Optional[Path] = False,
        metric_callback: Optional[UniversalStanceArgumentMetric] = None
) -> Dict[str, Dict[str, Tuple[bool, List[str]]]]:
    """
    Processes the given instances with the given ChatGPT-model and returns a dict of predictions.

    :param api_key: your OpenAI-API key
    :param task_prompt: the initial task prompt (which task should be executed?)
    :param instances: all instances to process
    :param users: a dict of users (user_id -> UserSelfReport) to use for decrypt the user ids in the instances
    :param model: the model string to use (see https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo and
    https://platform.openai.com/docs/models/gpt-3-5)
    :param required_user_attributes: user attributes that has to be requored in order to ask ChatGPT. If an int is
    given, the user has to have at least this number of attributes. If a set of strings is given, the user has to have
    all of them.
    :param pos_shots: the number of positive examples to use. If True, all positive shots are used. If False,
    no positive shots are used. If an int is given, the given number of positive shots are used.
    :param neg_shots: the number of negative examples to use. If True, all negative shots are used. If False,
    no negative shots are used. If an int is given, the given number of negative shots are used.
    :param text_target: the text target to use (see ExternOpinionArgument.to_text)
    :param arguments_per_instance: the number of arguments to generate per instance
    (for cherry-picking in post-processing)
    :param additional_chat_settings: additional settings for the ChatGPT-API (see
    https://platform.openai.com/docs/api-reference/chat/create) - optional
    :param save_to_file: if not None, the final dict is saved to this file
    :param metric_callback: a callback class-instance that is called after each instance is processed. The callback is
    optional - if not given, no metrics are computed (but a logged Stance-Classification-Report).
    :return: a dict of predictions (topic-> user -> (stance, [argument]))
    """
    logger.info("Let's get started! {} instances to process.", len(instances))

    openai.api_key = api_key
    logger.debug("OpenAI-API key set: {}", api_key)

    ret = defaultdict(dict)

    for instance in tqdm(instances, desc="Asking ChatGPT", unit="instance"):
        logger.trace("Processing instance: {}", instance.anchor)

        if instance.anchor.user not in users:
            logger.warning("User \"{}\" not found in user self-reports - skip {}!",
                           instance.anchor.user, instance.anchor)
            continue

        if required_user_attributes is not None:
            logger.trace("First, let's check whether the user \"{}\" is worthy...", instance.anchor.user)
            given_attributes = set()
            for attribute_string, attribute in [
                ("political orientation", users[instance.anchor.user].political_spectrum),
                ("relationship status", users[instance.anchor.user].relationship),
                ("gender", users[instance.anchor.user].gender),
                ("birthday", users[instance.anchor.user].birthday),
                ("education level", users[instance.anchor.user].education_level),
                ("ethnicity", users[instance.anchor.user].ethnicity),
                ("income", users[instance.anchor.user].income),
                ("working place", users[instance.anchor.user].working_place),
                ("religion", users[instance.anchor.user].religious)
            ]:
                if attribute in ["- Private -", "Not Saying", "Prefer not to say"]:
                    logger.trace("User \"{}\" hides his/her {}!", instance.anchor.user, attribute_string)
                else:
                    given_attributes.add(attribute_string)
            if isinstance(required_user_attributes, int):
                if len(given_attributes) < required_user_attributes:
                    logger.debug("User \"{}\" is not worthy (only {} known attributes) - skip {}!",
                                 instance.anchor.user, len(given_attributes), instance.anchor)
                    continue
            else:
                if not given_attributes.issuperset(required_user_attributes):
                    logger.debug("User \"{}\" is not worthy (not all required attributes are given) - skip {}!",
                                 instance.anchor.user, instance.anchor)
                    continue

        messages = [{"role": "user", "content": task_prompt}]
        for opinion_list, user_preference, string in [
            (instance.neg_instances, neg_shots, "negative"),
            (instance.pos_instances, pos_shots, "positive"),
            ([instance.anchor], True, "task")
        ]:
            if user_preference:
                limit = len(opinion_list) if isinstance(user_preference, bool)\
                    else min(len(opinion_list), user_preference)
                logger.trace("Adding {} {} shots", limit, string)
                for shot in opinion_list[:limit]:
                    try:
                        user = users[shot.user]
                        messages.append({
                            "role": "user",
                            "content": f"""Person {(user_short := shot.user.strip(" /"))[:min(len(user_short), 4)]} [
political orientation: {user.political_spectrum},
relationship status: {user.relationship},
gender: {user.gender},
birthday: {user.birthday},
education level: {user.education_level},
ethnicity: {user.ethnicity},
income: {user.income},
working place: {user.working_place},
religion: {user.religious}
]
Question: {shot.question_text}"""})

                        if string != "task":
                            messages.append({"role": "assistant", "content": f"""Stance: {shot.stance.upper()}
    Argument: {shot.to_text(target=text_target)}"""})
                    except KeyError:
                        logger.opt(exception=True).warning("User \"{}\" not found in user self-reports - skip {}!",
                                                           shot.user, shot)

        logger.debug("Sending {} messages to ChatGPT", len(messages))

        while True:
            try:
                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    n=arguments_per_instance,
                    stop=["\n\n", "Assistant:", "User:"],
                    **(additional_chat_settings or dict())
                )
                logger.trace("Successful ChatGPT {} ({}ms)", completion.api_version, completion.response_ms)
                logger.trace("Model \"{}\" processes {} tokens", completion.model, completion.usage["total_tokens"])

                stances = []
                arguments = []
                for choice in completion.choices:
                    logger.trace("ChatGPT choice #{}: {}", choice.index, choice.message)
                    if choice.message.content.startswith("Stance:"):
                        stances.append(int("YES" in choice.message.content.split("\n")[0].upper()))
                        if "\n" in choice.message.content:
                            arg = choice.message.content.split("\n")[1].strip(" \n\t")
                            if arg.startswith("Argument:"):
                                arguments.append(arg[len("Argument:"):].strip())
                            else:
                                arguments.append(arg)
                        else:
                            logger.info("ChatGPT {}. choice is not proper parsable, no argument here: {}",
                                        choice.index, choice.message.content)
                            arguments.append("YES" if stances[-1] else "NO")
                    else:
                        logger.debug("Unexpected: {} -- try to parse anyway", choice.message)
                        if choice.message.content.startswith("YES") or choice.message.content.startswith("NO"):
                            stances.append(int(choice.message.content.startswith("YES")))
                            arguments.append(
                                choice.message.content if "\n" not in choice.message.content
                                else " ".join(choice.message.content.split("\n")[1:])
                            )
                        else:
                            logger.warning("ChatGPT {}. choice is not proper parsable: {}",
                                           choice.index, choice.message.content)
                ret[instance.anchor.question_text][instance.anchor.user] = (
                    sum(stances) / len(stances),
                    arguments
                )

                logger.debug("Successful predicted {} (predicted PRO: {}%)", instance.anchor,
                             str(round(ret[instance.anchor.question_text][instance.anchor.user][0]*100)))
                break
            except openai.error.OpenAIError:
                logger.opt(exception=True).warning("Retrying...")
            except ZeroDivisionError:
                logger.opt(exception=True).warning("No useful response from ChatGPT!")
                if random.random() < .5:
                    logger.opt(exception=False).info("Retrying...")
                    continue
                logger.error("Skipping (seemly) unresolvable instance: {}", instance.anchor)
                ret[instance.anchor.question_text][instance.anchor.user] = (
                    0, ["ChatGPT-Failure", instance.anchor.question_text]
                )
                break
            logger.trace("Loop to \"{}\"", instance.anchor.to_text(target=text_target))

    logger.success("Done! ({} samples, {} topics)", len(instances), len(ret))

    final_dict = {"_end": {"test_inference": {
        "model": model,
        "task_prompt": task_prompt,
        "pos_shots": pos_shots,
        "neg_shots": neg_shots,
        "text_target": text_target,
        "arguments_per_instance": arguments_per_instance,
        "additional_chat_settings": additional_chat_settings or "None",
        "num_instances": len(instances),
        "stances_pred": [int(ret[instance.anchor.question_text][instance.anchor.user][0] > .5) for instance in instances
                         if instance.anchor.question_text in ret and
                         instance.anchor.user in ret[instance.anchor.question_text]],
        "stances_true": [int(instance.anchor.stance.upper() == "YES") for instance in instances
                         if instance.anchor.question_text in ret and
                         instance.anchor.user in ret[instance.anchor.question_text]],
        "predictions": ret,
        "ground_truth": {instance.anchor.question_text:
                             {user_reply.anchor.user: (int(user_reply.anchor.stance.upper() == "YES"),
                                                       user_reply.anchor.to_text(target=text_target))
                              for user_reply in instances
                              if user_reply.anchor.question_text == instance.anchor.question_text}
                         for instance in instances
                         if instance.anchor.question_text in ret and
                         instance.anchor.user in ret[instance.anchor.question_text]},
    }}}

    if metric_callback is not None:
        logger.debug("Computing metrics...")
        final_dict["_end"]["test_inference"]["metrics"] = (
            metric_callback(target=final_dict["_end"]["test_inference"]["ground_truth"],
                            predictions=final_dict["_end"]["test_inference"]["predictions"],
                            stance_threshold=.5))
        logger.info("Computed metrics: {} scores: {}...",
                    len(final_dict["_end"]["test_inference"]["metrics"]),
                    ", ".join((kl := list(final_dict["_end"]["test_inference"]["metrics"].keys()))[:min(5, len(kl))]))

    logger.success("Composed a final dict\nSTANCES\n{}",
                   classification_report(
                       y_true=final_dict["_end"]["test_inference"]["stances_true"],
                       y_pred=final_dict["_end"]["test_inference"]["stances_pred"],
                       target_names=["CON", "PRO"],
                       labels=[0, 1],
                       digits=3,
                       output_dict=False,
                       zero_division=0
                   ))

    if save_to_file is not None:
        logger.debug("Saving to file: {}", save_to_file)
        if save_to_file.exists():
            logger.warning("File {} already exists - appending!", save_to_file.absolute())

        if not save_to_file.name.endswith(".json"):
            save_to_file = save_to_file.joinpath("stats.json")

        save_to_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with save_to_file.open(mode="a", encoding="utf-8", errors="replace") as f:
                json_dump(obj=final_dict, fp=f, indent=4, sort_keys=True)
        except IOError:
            logger.opt(exception=True).error("Could not save to file: {}", save_to_file.absolute())
            logger.info(json_dumps(obj=final_dict, indent=4, sort_keys=False))

    return final_dict["_end"]["test_inference"]
