from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Any, Dict

from loguru import logger

import pandas


class CherryPickerInterface(ABC):
    def __init__(self, root_path: Path):
        logger.debug("Initializing a cherry-picker at the root-path \"{}\"", root_path.absolute())

        if root_path.exists():
            if root_path.is_dir():
                logger.info("Crawl through dir \"{}\"", root_path.name)
                path_dataframe_dict: Dict[Path, pandas.DataFrame] = {
                    csv: pandas.read_csv(filepath_or_buffer=csv, encoding="utf-8", index_col="User")
                    for csv in root_path.rglob(pattern="*.csv") if csv.is_file()
                }
                logger.success("Found {} topics!", len(path_dataframe_dict))
                logger.debug("\n".join(map(lambda k: "- {}".format(k.stem), path_dataframe_dict.keys())))
            elif root_path.is_file():
                logger.warning("Instead of getting a directory, you gave a single file: {} "
                               "(we process only this topic)", root_path.name)
                self.path_dataframe_dict: Dict[Path, pandas.DataFrame] = {
                    root_path: pandas.read_csv(filepath_or_buffer=root_path, encoding="uft-8", index_col="User")
                }
            else:
                logger.error("Strange path here, having a {}. Please point to an existing directory!",
                             root_path.suffix)
                self.path_dataframe_dict: Dict[Path, pandas.DataFrame] = dict()
        else:
            logger.warning("\"{}\" doesn't exist! Please execute the main method (Trainer) first!",
                           root_path.absolute())
            self.path_dataframe_dict: Dict[Path, pandas.DataFrame] = dict()

    @abstractmethod
    def cherry_pick(self, user: Any, topic: str, stance_probability: float, candidates: List[str]) -> int:
        """
        Cherry-picks the best candidate of all candidates given the author, the topic and the predicted stance
        probability. No ground-truth-values given here to not spoil the prediction!
        :param user: the user name (the author -- candidate should match the writing style and opinion of that user)
        :param topic: the topic (without special characters)
        :param stance_probability: the predicted stance that the user has to that topic
        (0: CON (for sure) -- 1: PRO (for sure))
        :param candidates: the generated candidates (arguments)
        :return: the index of the most proper candidate
        """
        pass

    def cherry_pick_all(self, save_cherries: bool = True) -> Dict[str, Dict[str, str]]:
        """
        Cherry-picks all the files in given root-dir using the cherry_pick-method
        :param save_cherries: if set to True, all cherry-picked files will be updated,
        containing the picked cherries then
        :return: the cherries in a dictionary: {topic: {user: best-generated-argument, ..}, ...}
        """
        ret = defaultdict(dict)
        for path, dataframe in self.path_dataframe_dict.items():
            path: Path
            dataframe: pandas.DataFrame
            logger.debug("Processing \"{}\"...", path.stem)
            selected_args = []
            for user, stuff in dataframe.iterrows():
                logger.trace("\"{}\"->{}", path.stem, user)
                candidates = [c for k, c in stuff.items() if "Pred_Arg" in k]
                if len(candidates) >= 2:
                    index = self.cherry_pick(
                        user=user,
                        stance_probability=stuff.get("Pred_Stance_PRO", .5),
                        topic=path.stem,
                        candidates=candidates
                    )
                    logger.debug("Cherry-picker selected No {}: {}", index, candidates[index])
                    selected_args.append(candidates[index])
                    ret[path.stem][user] = candidates[index]
                elif len(candidates) == 1:
                    logger.trace("Only one candidate, nothing to cherry-pick ({})", candidates[0])
                    selected_args.append(candidates[0])
                    ret[path.stem][user] = candidates[0]
                else:
                    logger.warning("No candidates for user {} (columns: {})!",
                                   user, ", ".join(map(lambda c: str(c), dataframe.columns)))
                    selected_args.append("n/a")
            logger.info("Successfully processed \"{}\", having {} selected arguments", path.name, len(selected_args))
            if save_cherries:
                logger.debug("Let's save the stuff in {}", path)
                column = "cherry ({})".format(self)
                if column in dataframe.columns:
                    logger.warning("There was already a cherry-picker here! Deleted the previous \"{}\"", column)
                    dataframe = dataframe.drop(columns=[column], inplace=False)
                else:
                    dataframe = dataframe.copy(deep=False)

                logger.trace("\"{}\" -> {}", column, selected_args)
                dataframe[column] = selected_args

                dataframe.to_csv(path_or_buf=path, encoding="utf-8", index=True, index_label="User")

        logger.success("Done with all {} files, collecting {} cherries",
                       len(self.path_dataframe_dict), sum(map(lambda v: len(v), ret.values())))

        return ret

    @abstractmethod
    def __str__(self) -> str:
        """
        Represents the cherry picker in a string. This is important for cherry_pick_all(self, save_cherries = True)
        since the new column containing the cherries is named using this method
        :return: the string representation
        """
        pass


class Ensemble(CherryPickerInterface):
    def __init__(self, root_path: Path, cherry_pickers: List[CherryPickerInterface]):
        super().__init__(root_path)

        self.cherry_pickers = cherry_pickers
        logger.debug("Select {} cherry-pickers: {}", len(self.cherry_pickers), self)

    def cherry_pick(self, user: Any, topic: str, stance_probability: float, candidates: List[str]) -> int:
        if len(self.cherry_pickers) == 0:
            logger.warning("No cherry picker available, pick the first candidate: \"{}\"", candidates[0])
            return 0

        votes = []
        for cherry_picker in self.cherry_pickers:
            logger.trace("Process {} candidates with {}", len(candidates), cherry_picker)
            votes.append(
                cherry_picker.cherry_pick(
                    user=user, topic=topic, stance_probability=stance_probability, candidates=candidates
                )
            )
            logger.trace("{} voted for \"{}\"", cherry_picker, candidates[votes[-1]])

        counter = Counter(votes)
        logger.debug("Gathered following votes: {}", counter.most_common())

        return counter.most_common(n=1)[0][0]

    def __str__(self) -> str:
        return "+".join(map(lambda cp: str(cp), self.cherry_pickers))
