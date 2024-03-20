import csv
import itertools
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import Iterable, Tuple, List, Any, Dict, Union, Optional, Literal, Iterator, Set

import torch
import transformers
from loguru import logger
from tqdm import tqdm

from pipeline.data_gathering.friends_graph.friends import Friends
from pipeline.data_gathering.opinion_arguments.ExternOpinionArgument import ExternOpinionArgument
from pipeline.data_gathering.opinion_arguments.Claim import Claim
from pipeline.data_gathering.users.UserSelfReport import UserSelfReport

from json import load as json_load, dump as json_dump

FRIENDSHIP_STRENGTH = 1.0

# path_to_poll_votes_file = os.path.join('data', 'poll_votes.csv')
path_to_extern_opinion_arguments_with_real_names_file = [
    Path('data', 'extern_opinion_arguments_with_real_names.csv'),
    Path('data', 'extern_additional_opinion_arguments_with_real_names.csv')
]
path_to_poll_votes_file_originally_with_binary_answers = Path('data',
                                                              'poll_votes_originally_with_binary_answers.csv')
path_to_poll_votes_file_originally_without_binary_answers = Path('data',
                                                                 'poll_votes_originally_without_binary_answers.csv')


def transform_to_dataloader(
        user_values_how_many_not_null: Optional[int] = None,
        load_additional_arguments: bool = True,
        verbose: bool = False
):

    # ---------------------------------------------------------------------------------------------------------------- #
    # parse data
    # parse extern opinion arguments
    extern_opinion_arguments = []
    topics = dict()
    for argument_file in path_to_extern_opinion_arguments_with_real_names_file[
                         :len(path_to_extern_opinion_arguments_with_real_names_file) if load_additional_arguments else -1]:
        with argument_file.open(newline='\n', encoding="utf-8") as csv_file_extern_opinion_arguments:
            extern_opinion_arguments_csv_rows = csv.reader(
                csv_file_extern_opinion_arguments, delimiter=';', quotechar='"'
            )  # "user";"link";"new_title";"new_vote";"explanation";"status";"sub-status";"original_title";"original_vote"

            # alternative name: claims
            for row in extern_opinion_arguments_csv_rows:  # argument_id, link, question_text, category, tags, user, stance, argument_title, text, conclusion
                extern_opinion_arguments.append(ExternOpinionArgument(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]))
                topics[Claim(row[1], row[2])] = None  # link, title

    # parse user
    with Path('data', f'user_with_num_of_not_null_entries_{user_values_how_many_not_null or 0}.csv').open(
            newline='\n', encoding="utf-8"
    ) as csv_file_users:
        users_csv_rows = csv.reader(csv_file_users, delimiter=';', quotechar='"')
        # "user_id";"political_spectrum";"relationship";"gender";"birthday";"current_age";"age_of_last_login";"education_level";"ethnicity";"income";"working_place";"religious";"number_of_debates"

        user_ids = set()
        user_properties = []
        for row in users_csv_rows:  # url, political_spectrum, relationship, gender, birthday, education_level, ethnicity, income, working_place, religious, number_of_debates, last_online
            user_properties.append(
                UserSelfReport(row[0], row[1], row[2], row[3], row[5], row[7],
                               row[8], row[9], row[10], row[11], row[12], row[6])
            )  # TODO: check whether row[5] is age instead of birthday and row[6] is age of last login instead of date
            user_ids.add(row[0])

    # The TODO was: :param user_ids: a list of all user ids which should sit in the jury (should give their opinions)
    # The TODO was: :param user_properties: the profiles of thw users

    # topics with batch (see https://stackoverflow.com/a/68338356)
    # The TODO was: :param topics: a list of all topics which should be discussed (or a string-list and an already performed batch encoding to save time)
    # embedder = SentenceTransformer('msmarco-distilbert-base-v2')  # TODO: Or another model
    # length_sorted_idx = np.argsort([-embedder._text_length(sen[1]) for sen in topics])
    # topics_sorted = [topics[idx] for idx in length_sorted_idx]

    arguments_per_topic_per_user = defaultdict(list)
    for extern_opinion_argument in extern_opinion_arguments:
        if extern_opinion_argument.user in user_ids:
            logger.log("DEBUG" if verbose else "TRACE", "User \"{}\" exists, keep opinion \"{}\"",
                       extern_opinion_argument.user, extern_opinion_argument.argument_title)
            arguments_per_topic_per_user[extern_opinion_argument.question_text].append(extern_opinion_argument)
        else:
            logger.log("INFO" if verbose else "TRACE", "User \"{}\" doesn't exist, discard opinion \"{}\"",
                       extern_opinion_argument.user, extern_opinion_argument)

    # parse friends_graph
    with Path('data', 'friends_graph.csv').open(newline='\n', encoding="utf-8") as csv_file_friends_graph:
        friends_graph_csv_rows = csv.reader(csv_file_friends_graph, delimiter=';', quotechar='"')  # "user_id";"friend_user_id"
        friends_graph = []
        for row in friends_graph_csv_rows:  # user_id_1, user_id_2
            friends_graph.append(Friends(row[0], row[1], FRIENDSHIP_STRENGTH))

    # ---------------------------------------------------------------------------------------------------------------- #

    return list(topics.keys()), user_ids, user_properties, arguments_per_topic_per_user, friends_graph


class ArgumentStanceDataset:
    class BatchIterator(Iterator):
        def __init__(
                self,
                polls: Dict[str, List[ExternOpinionArgument]],
                user_properties: List[UserSelfReport],
                text_target: Literal["argument_title", "conclusion", "premise", "whole_argument"],
                batch_size_topics: int,
                batch_size_users: int,
                sort_topics_by_length: bool,
                silent: bool = False
        ):
            # self.topic_link_to_string: Dict[str, str] = {c.link: c.title for c in topics}
            self.user_properties = {
                user_property.url: {p: v for p, v in user_property.__dict__.items() if p != "url"}
                for user_property in user_properties
            }
            self.polls = polls
            if any(map(lambda poll_list: len(poll_list) == 0 or
                                         any(map(lambda poll: poll.user not in self.user_properties, poll_list)),
                       self.polls.values())):
                raise ValueError("There are empty polls out there, Or polls referring to non-existing user!")

            self.topics = list(self.polls.keys())
            if sort_topics_by_length:
                self.topics.sort(key=lambda t: len(t), reverse=True)
            else:
                self.topics.sort(key=lambda t: len(self.polls[t]), reverse=True)
            self.tensor_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.batch_size_topics = batch_size_topics
            self.batch_size_users = batch_size_users
            self.text_target = text_target
            self.silent = silent

        def __next__(self):
            if len(self.topics) == 0:
                logger.info("Crawled through all topics, {} DONE", len(self.polls))
                raise StopIteration

            def user_intersection(f_selected_topics: List[str]) -> Set[Any]:
                if not self.silent:
                    logger.trace("Determine the user intersection of {} topics", len(f_selected_topics))
                if len(f_selected_topics) == 0:
                    return set()

                set_selected_topics = set(f_selected_topics)
                set_users = {pv.user for pv in self.polls[set_selected_topics.pop()]}
                while len(set_selected_topics) >= 1 and len(set_users) >= 1:
                    set_users.intersection_update({pv.user for pv in self.polls[set_selected_topics.pop()]})

                if not self.silent:
                    logger.debug("All {} topics have {} users in common: {}", len(f_selected_topics), len(set_users),
                                 ", ".join(map(lambda u: "\"{}\"".format(u), set_users)))
                return set_users

            selected_topics = self.topics[:self.batch_size_topics]
            selected_users = user_intersection(f_selected_topics=selected_topics)
            try:
                while len(selected_users) == 0:
                    removed_topic = selected_topics.pop()
                    if not self.silent:
                        logger.debug("No shared users in these {} topics- we have to remove the topic {}",
                                     len(selected_topics)+1, removed_topic)
                    selected_users = user_intersection(f_selected_topics=selected_topics)
            except IndexError:
                logger.opt(exception=True).log("WARNING" if self.silent else "ERROR",
                                               "Got an empty topic {} probably", selected_topics)

            listed_selected_topics = list(selected_topics)
            users = list(selected_users)[:min(len(selected_users), self.batch_size_users)]
            if not self.silent:
                logger.trace("Finally selected {} topics and {} users", len(listed_selected_topics), len(users))
            user_properties = [self.user_properties[u] for u in users]
            if self.tensor_device == "skip":
                if not self.silent:
                    logger.info("Skip the computation of the ground-truth-values since \"tensor_device\"=\"{}\"",
                                self.tensor_device)
                ground_truth_stances_per_topic_per_user = None
                ground_truth_arguments_per_topic_per_user = None
            else:
                ground_truth_stances_per_topic_per_user = torch.tensor(
                    data=[[[int(poll.stance == "yes") for poll in self.polls[topic] if poll.user == user][0]
                           for user in users] for topic in listed_selected_topics],
                    dtype=torch.long,
                    device=self.tensor_device
                )
                ground_truth_arguments_per_topic_per_user = \
                    [[[poll.to_text(target=self.text_target) for poll in self.polls[topic] if poll.user == user][0]
                      for user in users] for topic in listed_selected_topics]
                if any(map(lambda user_level: any(map(lambda arg: isinstance(arg, torch.Tensor), user_level)),
                           ground_truth_arguments_per_topic_per_user)):
                    if not self.silent:
                        logger.trace("Get the ground_truth_arguments_per_topic_per_user on the desired device: \"{}\"",
                                     self.tensor_device)
                    ground_truth_arguments_per_topic_per_user = torch.stack(
                        tensors=[torch.stack(tensors=user_level, dim=0)
                                 for user_level in ground_truth_arguments_per_topic_per_user],
                        dim=0
                    ).to(self.tensor_device)

            topics_to_append = []
            for topic in listed_selected_topics:
                for user in users:
                    if not self.silent:
                        logger.trace("User \"{}\" is proceeded in topic {}", user, topic)
                    self.polls[topic].remove([poll for poll in self.polls[topic] if poll.user == user][0])
                if len(self.polls[topic]) >= 1:
                    topics_to_append.append(topic)
                    if not self.silent:
                        logger.trace("Topic \"{}\" is has still {} queued users", topic, len(self.polls[topic]))
                elif not self.silent:
                    logger.debug("Topic \"{}\" is done", topic)

            if not self.silent:
                logger.info("One batch is packed. From the {} returned topics, {} of them have still queued users",
                            len(listed_selected_topics), len(topics_to_append))
            self.topics = self.topics[len(listed_selected_topics):] + topics_to_append

            return (listed_selected_topics, users, user_properties,
                    ground_truth_stances_per_topic_per_user, ground_truth_arguments_per_topic_per_user)

    def __init__(
            self,
            tokenizer: Optional[str] = None,
            header_in_data: bool = True,
            limit_data: Optional[int] = None,
            user_values_how_many_not_null: Optional[int] = None,
            load_additional_arguments: bool = True,
            train_dev_test_splitting: Optional[Tuple[float, float, float]] = None,
            train_dev_test_splitting_format:
            Literal["overlap_user_topic", "overlap_user", "overlap_topic", "avoid_overlap"] = "overlap_user",
            shuffle_data: bool = True,
            text_target: Literal["argument_title", "conclusion", "premise", "whole_argument"] = "conclusion",
            verbose: bool = False
    ):
        self.verbose = verbose
        self.text_target = text_target

        logger.trace("OK, we have to load the data...")
        data = transform_to_dataloader(
            user_values_how_many_not_null=user_values_how_many_not_null,
            load_additional_arguments=load_additional_arguments,
            verbose=self.verbose
        )
        logger.info("Successfully fetched the data with {} entries",
                    " and ".join(map(lambda d: "{:,}".format(len(d)), data)))

        self.topics: List[Claim] = data[0][
                                   int(header_in_data):
                                   min(
                                       len(data[0]), limit_data+int(header_in_data) if limit_data is not None else
                                       len(data[0])
                                   )]
        logger.debug("{} topics", len(self.topics))

        self.user_ids: List[str] = list(data[1])[int(header_in_data):]
        logger.debug("{} users", len(self.user_ids))
        self.user_properties: List[UserSelfReport] = data[2][int(header_in_data):]

        self.grouped_opinions: Dict[str, List[ExternOpinionArgument]] = data[3]
        if header_in_data and "question_text" in self.grouped_opinions:
            logger.debug("Remove the head {}", self.grouped_opinions.pop("question_text"))
        if limit_data is not None:
            title_topics = {t.title for t in self.topics}
            for topic in list(self.grouped_opinions.keys()):
                if topic not in title_topics:
                    logger.log("INFO" if self.verbose else "TRACE",
                               "Remove topic \"{}\" with {} participants because it was truncated (limit_data={})",
                               topic, len(self.grouped_opinions.pop(topic)), limit_data)
        if any(map(lambda gpv: len(gpv) == 0, self.grouped_opinions.values())):
            logger.warning("There are empty polls out there, we have to filter them out!")
            for topic_link, participants in self.grouped_opinions.items():
                if len(participants) == 0:
                    logger.log("INFO" if self.verbose else "TRACE", "Poll \"{}\" is empty: {}!",
                               topic_link, self.grouped_opinions.pop(topic_link))
                    self.topics.remove([t for t in self.topics if t.link == topic_link][0])
                else:
                    logger.log("DEBUG" if self.verbose else "TRACE", "Topic \"{}\" has {} participants, all ok :)",
                               len(participants))
        logger.debug("{} poll votes, ({} participants on average)",
                     len(self.grouped_opinions),
                     str(round(sum(map(lambda v: len(v), self.grouped_opinions.values())) /
                               len(self.grouped_opinions), 1)))

        self.friends_graph: List[Friends] = data[4][int(header_in_data):]
        if len(self.friends_graph) >= 1:
            logger.log("WARNING" if self.verbose else "TRACE",
                       "Ignore the friendship-graph ({} friendships), "
                       "we have to write the friendships into the config.json", len(self.friends_graph))

        if tokenizer is not None:
            logger.debug("OK, first we tokenize ;)")
            self.tokenizer: transformers.PreTrainedTokenizer = \
                transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer)
            logger.info("Successfully loaded the tokenizer \"{}\"", self.tokenizer.name_or_path)

            for poll_votes in self.grouped_opinions.values():
                for poll_vote in poll_votes:
                    poll_vote.set_vector_representation(
                        for_target=text_target,
                        vector_representation=self.tokenizer.encode(text=poll_vote.to_text(target=text_target),
                                                                    return_tensors="pt")
                    )

        logger.info("OK, now we have to split the data into train, dev and test")
        if train_dev_test_splitting is None:
            train_dev_test_splitting = (.8, .1, .1)
        elif sum(train_dev_test_splitting) != 1:
            logger.warning("Your train_dev_test_splitting ({}) has to be normalized (divided by {})",
                           train_dev_test_splitting, sum(train_dev_test_splitting))
            train_dev_test_splitting = tuple(t/sum(train_dev_test_splitting) for t in train_dev_test_splitting)
        logger.debug(
            "OK, consider the splitting: {}",
            "-".join(map(lambda p: "{}%".format(str(round(p * 100))), train_dev_test_splitting))
        )

        splitting_path = Path(
            ".out", "data-splits",
            f"{train_dev_test_splitting_format} ({'-'.join(map(str, train_dev_test_splitting))})",
            "shuffled" if shuffle_data else "unshuffled",
            f"limit_data={limit_data}" if limit_data is not None else
            f"{'all' if load_additional_arguments else 'main'}_data",
            f"min{user_values_how_many_not_null}-attr_users.split" if user_values_how_many_not_null is not None else
            "all_users.split"
        )
        logger.debug("Checking \"{}\" whether the data is already split", splitting_path)
        if splitting_path.exists():
            logger.debug("Load the splitting from \"{}\"", splitting_path.absolute())
            with splitting_path.open(mode="r", encoding="utf-8") as splitting_filestream:
                id_splits: Dict[str, List[str]] = json_load(fp=splitting_filestream)
            logger.info("Load {} splits from \"{}\" ({} total)", len(id_splits),
                        splitting_path.name, sum(map(len, id_splits.values())))
            dict_opinion_args: Dict[str, ExternOpinionArgument] \
                = {ext.argument_id: ext for ext in itertools.chain.from_iterable(self.grouped_opinions.values())}
            self.splits: Dict[str, List[ExternOpinionArgument]] = {
                split: [dict_opinion_args[arg_id] for arg_id in arg_ids] for split, arg_ids in id_splits.items()
            }
        else:
            train_dev_test_splitting = (*train_dev_test_splitting, 0)

            self.splits: Dict[str, List[ExternOpinionArgument]] = defaultdict(list)
            if train_dev_test_splitting_format == "overlap_user_topic" or train_dev_test_splitting_format == "overlap_topic":
                for list_opinions in tqdm(iterable=self.grouped_opinions.values(), desc="Splitting the data", unit="topic",
                                          total=len(self.grouped_opinions)):
                    logger.debug("OK, process {} opinions for one topic, trying to {}",
                                 len(list_opinions), train_dev_test_splitting_format)
                    if shuffle_data:
                        random.shuffle(list_opinions)
                    if train_dev_test_splitting_format == "overlap_user_topic":
                        for i, split in enumerate(["train", "dev", "test"]):
                            self.splits[split].extend(
                                list_opinions[
                                int(sum(train_dev_test_splitting[:i]) * len(list_opinions)):
                                int(sum(train_dev_test_splitting[:i+1]) * len(list_opinions))
                                ])
                    else:
                        logger.trace("We have to avoid an overlap of users in the splits")
                        open_list_opinions = list_opinions.copy()
                        users_in_splits = [
                            (split, {opinion.user for opinion in opinion_list}) for split, opinion_list in
                            [("train", self.splits["train"]), ("dev", self.splits["dev"]), ("test", self.splits["test"])]
                        ]
                        user_duplicate_counter = Counter([o.user for o in list_opinions])
                        if user_duplicate_counter.most_common(n=1)[0][1] > 1:
                            list_opinions = list_opinions.copy()
                            logger.warning("One or more users voted twice or more, we have to remove them (e.g.: {})",
                                           user_duplicate_counter.most_common(n=1))
                            for user, count in user_duplicate_counter.most_common(n=None):
                                if count > 1:
                                    double_opinions = [opinion for opinion in list_opinions if opinion.user == user]
                                    logger.info("User \"{}\" voted {} times, we have to remove {} of them // {}",
                                                user, count, count - 1, "/".join(map(str, double_opinions)))
                                    for opinion in double_opinions[int(not shuffle_data):]:
                                        list_opinions.remove(opinion)
                                    if shuffle_data:
                                        list_opinions.append(random.sample(double_opinions, 1)[0])
                                else:
                                    logger.trace("No more duplicate users")
                                    break
                        for opinion in list_opinions:
                            user_appears_in_splits = [split for split, users in users_in_splits
                                                      if opinion.user in users]
                            if len(user_appears_in_splits) >= 1:
                                logger.trace("User \"{}\" appears in the splits {}, "
                                             "we have to put the opinion \"{}\" in that split",
                                             opinion.user, user_appears_in_splits, opinion)
                                self.splits[user_appears_in_splits[0]].append(opinion)
                                open_list_opinions.remove(opinion)
                        logger.debug("OK, we have {} (-{}) opinions left, we can split them now",
                                     len(open_list_opinions), len(list_opinions) - len(open_list_opinions))
                        for i, split in enumerate(["train", "dev", "test"]):
                            self.splits[split].extend(
                                open_list_opinions[
                                int(sum(train_dev_test_splitting[:i]) * len(open_list_opinions)):
                                int(sum(train_dev_test_splitting[:i+1]) * len(open_list_opinions))
                                ])
            elif train_dev_test_splitting_format == "overlap_user" or train_dev_test_splitting_format == "avoid_overlap":
                if train_dev_test_splitting_format == "overlap_user":
                    logger.debug("OK, we have to avoid an overlap of topics ({}) in the splits", len(self.topics))
                else:
                    logger.debug("OK, we have to avoid an overlap of topics ({}) and  users ({}) in the splits",
                                 len(self.topics), len(self.user_ids))
                listed_grouped_opinions = [opinion_list for opinion_list in self.grouped_opinions.values()]
                if shuffle_data:
                    random.shuffle(listed_grouped_opinions)
                for i, split in enumerate(["train", "dev", "test"]):
                    self.splits[split].extend(
                        listed_grouped_opinions[
                                int(sum(train_dev_test_splitting[:i]) * len(listed_grouped_opinions)):
                                int(sum(train_dev_test_splitting[:i+1]) * len(listed_grouped_opinions))
                                ]
                    )
                logger.debug("Split the topics non-overlapping into {} train topics, {} dev topics and {} test topics",
                             len(self.splits["train"]), len(self.splits["dev"]), len(self.splits["test"]))
                self.splits: Dict[str, List[ExternOpinionArgument]] = {
                    split: list(itertools.chain(*list_of_opinion_lists))
                    for split, list_of_opinion_lists in self.splits.items()
                }
                logger.debug("Split the opinions non-overlapping into {} train opinions, "
                             "{} dev opinions and {} test opinions",
                             len(self.splits["train"]), len(self.splits["dev"]), len(self.splits["test"]))
                if train_dev_test_splitting_format == "avoid_overlap":
                    logger.debug("OK, we have to avoid an overlap of users in the splits")
                    users_in_splits: Dict[str, Set[str]] = {
                        split: {opinion.user for opinion in opinion_list} for split, opinion_list in self.splits.items()
                    }
                    users_should_be_in_split = {
                        "test": users_in_splits["test"],
                        "dev": users_in_splits["dev"].difference(users_in_splits["test"]),
                        "train": users_in_splits["train"].difference(users_in_splits["test"]).difference(users_in_splits["dev"])
                    }
                    logger.debug("OK, we have {} train users, {} dev users and {} test users",
                                 len(users_should_be_in_split["train"]), len(users_should_be_in_split["dev"]),
                                 len(users_should_be_in_split["test"]))
                    non_user_overlapping_splits = {
                        split: [opinion for opinion in opinion_list if opinion.user in users_should_be_in_split[split]]
                        for split, opinion_list in self.splits.items()
                    }
                    for split in self.splits.keys():
                        if len(self.splits[split]) > len(non_user_overlapping_splits[split]):
                            logger.warning("We have to remove {} opinions from the split \"{}\""
                                           "to ensure non overlapping users",
                                           len(self.splits[split]) - len(non_user_overlapping_splits[split]), split)
                            self.splits[split] = non_user_overlapping_splits[split]
                        else:
                            logger.debug("OK, the split \"{}\" has still {} opinions, all ok :)",
                                         split, len(non_user_overlapping_splits[split]))
            else:
                raise ValueError("Unhandled train_dev_test_splitting_format: {}".format(train_dev_test_splitting_format))

            logger.debug("OK, we have to write the splitting into \"{}\" for later use", splitting_path.absolute())

            splitting_path.parent.mkdir(parents=True, exist_ok=True)
            with splitting_path.open(mode="w", encoding="utf-8") as splitting_filestream:
                json_dump(
                    obj={split: [opinion.argument_id for opinion in opinion_list]
                         for split, opinion_list in self.splits.items()},
                    fp=splitting_filestream,
                    sort_keys=False,
                    indent=None
                )
            logger.info("Successfully wrote the splitting into \"{}\"", splitting_path)

        logger.success("Split the data in {} train instances, {} dev instances and {} test instances",
                       self.num_instances(split="train"), self.num_instances(split="dev"),
                       self.num_instances(split="test"))

    def num_instances(self, split: Literal["train", "dev", "test"]) -> int:
        return len(self.splits[split])

    def num_batches(
            self,
            split: Literal["train", "dev", "test"],
            batch_size_topics: int,
            batch_size_users: int,
            sort_topics_by_length: bool
    ) -> int:
        polls = defaultdict(list)
        for opinion in self.splits[split]:
            polls[opinion.question_text].append(opinion)

        iterator = ArgumentStanceDataset.BatchIterator(
            polls=polls,
            user_properties=self.user_properties,
            batch_size_topics=batch_size_topics,
            batch_size_users=batch_size_users,
            sort_topics_by_length=sort_topics_by_length,
            text_target=self.text_target,
            silent=True
        )
        iterator.tensor_device == "skip"
        return len(list(iterator))

    def get_iterable(
            self,
            split: Literal["train", "dev", "test"],
            batch_size_topics: int,
            batch_size_users: int,
            sort_topics_by_length: bool,
            tensors_device: Optional[str] = None
    ) -> Iterable[Tuple[
        List[str],
        List[Any],
        List[Dict[str, str]],
        torch.LongTensor,
        Union[List[List[str]], torch.LongTensor]
    ]]:
        polls = defaultdict(list)
        for opinion in self.splits[split]:
            polls[opinion.question_text].append(opinion)

        iterator = ArgumentStanceDataset.BatchIterator(
            polls=polls,
            user_properties=self.user_properties,
            batch_size_topics=batch_size_topics,
            batch_size_users=batch_size_users,
            sort_topics_by_length=sort_topics_by_length,
            text_target=self.text_target,
            silent=not self.verbose
        )
        if tensors_device is not None:
            iterator.tensor_device = tensors_device

        return iterator
