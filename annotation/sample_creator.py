import csv
import json
import os.path
import random
import re

from annotation.EvaluationEntry import EvaluationEntry

path_to_avoidOverlap_zeroShots = os.path.join('..', '.out', 'prompting', 'main', 'avoid-overlap', 'zero-shot-3.json')
path_to_avoidOverlap_positiveFewShots = os.path.join('..', '.out', 'prompting', 'main', 'avoid-overlap', '3-shot-3.json')
path_to_avoidOverlap_positiveNegativeFewShots = os.path.join('..', '.out', 'prompting', 'main', 'avoid-overlap', '3-3-shot-3.json')

path_to_overlapTopic_positiveFewShots = os.path.join('..', '.out', 'prompting', 'main', 'overlap-topic', '3-shot-3.json')
path_to_overlapUser_positiveFewShots = os.path.join('..', '.out', 'prompting', 'main', 'overlap-user', '3-shot-3.json')
path_to_overlapUserTopic_positiveFewShots = os.path.join('..', '.out', 'prompting', 'main', 'overlap-user-topic', '3-shot-3.json')

# Von den 5 Runs für den fine-tuned-Ansatz nehmen wir Run 2, weil dieser die mittlere Performance beim stance_F1 abliefert.
path_to_fineTunedApproach = os.path.join('..', '.out', 'mainexperiments', 'avoid_overlap', 'controlledGEN-unbounded', 'large--complex', '_multirun_all-MiniLM-L12-v2--simple-friendship--big-knowledge-user-prob', 'Run-2', 'stats.json')


map_approachName_pathToApproach = {'avoidOverlap_zeroShots': path_to_avoidOverlap_zeroShots,
                                   'avoidOverlap_positiveFewShots': path_to_avoidOverlap_positiveFewShots,
                                   #'avoidOverlap_positiveNegativeFewShots': path_to_avoidOverlap_positiveNegativeFewShots,
                                   #'overlapTopic_positiveFewShots': path_to_overlapTopic_positiveFewShots,
                                   #'overlapUser_positiveFewShots': path_to_overlapUser_positiveFewShots,
                                   #'overlapUserTopic_positiveFewShots': path_to_overlapUserTopic_positiveFewShots,
                                   'fineTunedApproach': path_to_fineTunedApproach}


# buckets = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
buckets = [(0, 0), (1, 3), (4, 5), (6, 9)]
# buckets = [(0, 0), (1, 4), (5, 9)]
limit_of_questions_in_final_sample = 30
limit_of_users_in_final_sample = 5


def main():

    map_userId_minDimensions = map_user_ids_to_their_number_of_dimensions()
    map_userId_properties = map_user_ids_to_their_properties()

    map_bucket_question_user_approachName_evaluationEntry = generate_map_with_all_approaches_with_their_predictions(map_userId_minDimensions)
    sampleMap_bucket_question_user_approachName_evaluationEntry = generate_final_sample_for_annotation(map_bucket_question_user_approachName_evaluationEntry, limit_of_questions_in_final_sample, limit_of_users_in_final_sample)
    write_data_to_annotate(sampleMap_bucket_question_user_approachName_evaluationEntry, map_userId_minDimensions, map_userId_properties)


def map_user_ids_to_their_number_of_dimensions():
    map_userId_minDimensions = {}
    for minDimension in list(reversed(range(0, 10))):
        path_to_file_with_user_with_num_of_not_null_entries = os.path.join('..', 'data', 'user_with_num_of_not_null_entries_' + str(minDimension) + '.csv')

        with open(path_to_file_with_user_with_num_of_not_null_entries, newline='\n', encoding="utf8") as csv_file_read:
            csv_reader = csv.reader(csv_file_read, delimiter=';', quotechar='"')
            next(csv_reader, None)  # skip the headers

            for row in csv_reader:
                user_id = row[0]
                if user_id not in map_userId_minDimensions:
                    map_userId_minDimensions[user_id] = minDimension

            csv_file_read.flush()
            csv_file_read.close()
    return map_userId_minDimensions


def map_user_ids_to_their_properties():
    map_userId_properties = {}
    for data_split in ['avoid_overlap', 'overlap_topic', 'overlap_user', 'overlap_user_topic']:
        path_to_prompts = os.path.join('..', 'data', 'prompts_2023_12_03', 'prompts_for_chat_gpt_' + data_split + '_withLimitOf2000.csv')

        with open(path_to_prompts, newline='\n', encoding="utf8") as csv_file_read:
            csv_reader = csv.reader(csv_file_read, delimiter=';', quotechar='"')
            next(csv_reader, None)  # skip the headers

            for row in csv_reader:
                # "opinion_argument_id";"path_to_data_split";"minDimension";"max_number_of_examples_in_this_run";"test_user_id";"positive_or_negative_example";"similar_user_ids";"similar_topic_ids";"prompt_1_user";"prompt_2_user";"prompt_3plus_user"
                user_id = row[4]
                prompt_2 = row[9]
                property_matcher = re.search(r'(\[\s+((.|\n)*)\])', prompt_2)
                properties = property_matcher.group(1)

                if user_id not in map_userId_properties:
                    map_userId_properties[user_id] = properties

            csv_file_read.flush()
            csv_file_read.close()
    return map_userId_properties


def generate_map_with_all_approaches_with_their_predictions(map_userId_minDimensions):

    map_bucket_question_user_approachName_evaluationEntry = {}
    for bucket in buckets:
        map_bucket_question_user_approachName_evaluationEntry[str(bucket[0]) + '-' + str(bucket[1])] = {}

    for approachName in map_approachName_pathToApproach:
        with open(map_approachName_pathToApproach[approachName]) as json_file:

            json_data = json.load(json_file)['_end']['test_inference']

            ground_truth = json_data['ground_truth']
            for question in ground_truth:
                for user in ground_truth[question]:
                    ground_truth_stance = ground_truth[question][user][0]
                    ground_truth_answer = ground_truth[question][user][1]

                    bucket = get_bucket(map_userId_minDimensions, user)

                    if question not in map_bucket_question_user_approachName_evaluationEntry[bucket]:
                        map_bucket_question_user_approachName_evaluationEntry[bucket][question] = {}

                    if user not in map_bucket_question_user_approachName_evaluationEntry[bucket][question]:
                        map_bucket_question_user_approachName_evaluationEntry[bucket][question][user] = {}

                    # workaround for fine-tuned-approach
                    if approachName == 'fineTunedApproach':
                        continue

                    map_bucket_question_user_approachName_evaluationEntry[bucket][question][user][approachName] = EvaluationEntry(
                        user,
                        question,
                        approachName,
                        ground_truth_stance,
                        ground_truth_answer
                    )

            predictions = json_data['predictions']
            for question in predictions:
                for user in predictions[question]:
                    predicted_stance = predictions[question][user][0]
                    predicted_answer = predictions[question][user][1][0]  # W.r.t. our standard experiments, we ignore the cherrypicker and pick the first answer as prediciton

                    bucket = get_bucket(map_userId_minDimensions, user)

                    if approachName == 'fineTunedApproach':
                        map_bucket_question_user_approachName_evaluationEntry[bucket][question][user][
                            approachName] = EvaluationEntry(
                            user,
                            question,
                            approachName,
                            map_bucket_question_user_approachName_evaluationEntry[bucket][question][user]['avoidOverlap_zeroShots'].ground_truth_stance,
                            map_bucket_question_user_approachName_evaluationEntry[bucket][question][user]['avoidOverlap_zeroShots'].ground_truth_answer
                        )

                    map_bucket_question_user_approachName_evaluationEntry[bucket][question][user][approachName].predicted_stance = predicted_stance
                    map_bucket_question_user_approachName_evaluationEntry[bucket][question][user][approachName].predicted_answer = predicted_answer

            json_file.flush()
            json_file.close()

    return map_bucket_question_user_approachName_evaluationEntry


def get_bucket(map_userId_minDimensions, user):
    user_dimension = map_userId_minDimensions[user]
    for bucket in buckets:
        if bucket[0] <= user_dimension <= bucket[1]:
            return str(bucket[0]) + '-' + str(bucket[1])


def generate_final_sample_for_annotation(map_bucket_question_user_approachName_evaluationEntry, max_questions, max_users):

    count_all = 0

    sampleMap_bucket_question_user_approachName_evaluationEntry = {}
    for bucket in map_bucket_question_user_approachName_evaluationEntry:

        count_question = 0
        for question in map_bucket_question_user_approachName_evaluationEntry[bucket]:
            if count_question == max_questions:
                continue
            else:
                count_question += 1

            count_user = 0
            for user in map_bucket_question_user_approachName_evaluationEntry[bucket][question]:
                if count_user == max_users:
                    continue
                else:
                    count_user += 1

                for approachName in map_bucket_question_user_approachName_evaluationEntry[bucket][question][user]:

                    if bucket not in sampleMap_bucket_question_user_approachName_evaluationEntry:
                        sampleMap_bucket_question_user_approachName_evaluationEntry[bucket] = {}
                    if question not in sampleMap_bucket_question_user_approachName_evaluationEntry[bucket]:
                        sampleMap_bucket_question_user_approachName_evaluationEntry[bucket][question] = {}
                    if user not in sampleMap_bucket_question_user_approachName_evaluationEntry[bucket][question]:
                        sampleMap_bucket_question_user_approachName_evaluationEntry[bucket][question][user] = {}

                    sampleMap_bucket_question_user_approachName_evaluationEntry[bucket][question][user][approachName] = map_bucket_question_user_approachName_evaluationEntry[bucket][question][user][approachName]
                    count_all += 1

    print('Number of samples to annotate: ' + str(count_all) + ' = ' + str(max_questions) + " * " + str(max_users) + ' * ' + str(len(map_bucket_question_user_approachName_evaluationEntry)) + ' * ' + str(len(map_approachName_pathToApproach)))

    return sampleMap_bucket_question_user_approachName_evaluationEntry

def write_data_to_annotate(map_bucket_question_user_approachName_evaluationEntry, map_userId_minDimensions, map_userId_properties):
    overlap_count = 0
    id_count = 0
    questions_map = {}
    for bucket in map_bucket_question_user_approachName_evaluationEntry:
        for question in map_bucket_question_user_approachName_evaluationEntry[bucket]:
            for user in map_bucket_question_user_approachName_evaluationEntry[bucket][question]:
                overlap_exists = True

                sub_questions_map = {}

                shuffledApproachNames = list(map_approachName_pathToApproach.keys())
                random.shuffle(shuffledApproachNames)

                for approachName in shuffledApproachNames:

                    if approachName not in map_bucket_question_user_approachName_evaluationEntry[bucket][question][user]:
                        overlap_exists = False
                        break

                    evaluation_entry = map_bucket_question_user_approachName_evaluationEntry[bucket][question][user][approachName]

                    if (evaluation_entry.ground_truth_stance is None
                            or evaluation_entry.ground_truth_answer is None
                            or evaluation_entry.predicted_stance is None
                            or evaluation_entry.predicted_answer is None):
                        overlap_exists = False
                        break

                    sub_questions_map[id_count] = {
                        "question": question,
                        "user": user,
                        "stakeholder_group": map_userId_properties[user],
                        "predicted_stance": evaluation_entry.predicted_stance,
                        "ground_truth_stance": evaluation_entry.ground_truth_stance,
                        "ground_truth_explanation": evaluation_entry.ground_truth_answer,
                        "generated_explanation": evaluation_entry.predicted_answer,
                        "approach_name": approachName,
                        "minDimensions": map_userId_minDimensions[user],
                        "bucket": bucket
                    }
                    id_count += 1

                if overlap_exists:
                    overlap_count += 1

                    questions_map = questions_map | sub_questions_map


    print('topic-user-overlap: ' + str(overlap_count))
    with open('questions.json', 'w') as json_output_file:
        json.dump(questions_map, json_output_file)


if __name__ == '__main__':
    main()
