import csv
import json
import os

from pipeline.PromptChatGPT.preprocess.TrainDevTestSplit import TrainDevTestSplit
from pipeline.data_gathering.opinion_arguments.ExternOpinionArgument import ExternOpinionArgument


def get_map_with_paths_to_train_dev_test_splits():
    dir_path = os.path.abspath(os.path.join('..', '..', '..', '.out', 'data-splits'))
    data_splits = get_paths_to_data_splits_with_ids_to_opinion_arguments(dir_path)
    map_path_trainDevTestSplits = read_in_ids_of_opinion_arguments(data_splits)

    return map_path_trainDevTestSplits


def get_paths_to_data_splits_with_ids_to_opinion_arguments(dir_path):
    data_splits = []
    for (dir_path, dir_names, file_names) in os.walk(dir_path):
        for file_name in file_names:
            data_splits.append(os.path.join(str(dir_path), str(file_name)))
    return data_splits


def read_in_ids_of_opinion_arguments(data_splits):
    map_path_trainDevTestSplits = {}
    for data_split in data_splits:
        map_path_trainDevTestSplits[data_split] = []
        with open(data_split) as data_split_json:
            json_content = json.load(data_split_json)
            map_path_trainDevTestSplits[data_split] = TrainDevTestSplit(
                    json_content.get('train'),
                    json_content.get('dev'),
                    json_content.get('test')
                )
    return map_path_trainDevTestSplits


def generate_map_with_ids_to_opinion_arguments(data_split):
    map_opinionArgumentId_opinionArgument = {}
    paths_to_extern_opinion_arguments = [
        os.path.abspath(os.path.join('..', '..', '..', 'data', 'extern_opinion_arguments_with_real_names.csv')),
        os.path.abspath(os.path.join('..', '..', '..', 'data', 'extern_additional_opinion_arguments_with_real_names.csv'))]
    for path_to_extern_opinion_arguments in paths_to_extern_opinion_arguments:
        with open(path_to_extern_opinion_arguments, newline='\n', encoding="utf8") as csv_file_read:
            csv_reader = csv.reader(csv_file_read, delimiter=';', quotechar='"')
            next(csv_reader, None)  # skip the headers

            for row in csv_reader:

                if row[0] in data_split.train or row[0] in data_split.dev or row[0] in data_split.test:

                    # "argument_id";"link";"question_text";"category";"tags";"user";"stance";"argument_title";"text";"conclusion"
                    map_opinionArgumentId_opinionArgument[row[0]] = ExternOpinionArgument(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9])

    return map_opinionArgumentId_opinionArgument


def test_mapping_of_train_dev_test_splits(data_split, map_id_opinionArgument):
        for ids in [data_split.train, data_split.dev, data_split.test]:
            for id in ids:
                if map_id_opinionArgument[id] is None:
                    raise Exception('Error with id: ' + str(id))
