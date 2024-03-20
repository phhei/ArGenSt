import csv
import os
import re

from pipeline.PromptChatGPT.preprocess.Quadruple_UserId_TopicId_Prompt_Score import Quadruple_UserId_TopicId_Prompt_Score
from pipeline.PromptChatGPT.preprocess.step_3_most_similar_users_searcher import get_map_with_minDimensions_and_their_users


def generate_prompts(data_split, limit_of_similar_users_and_topics, map_userId_userSelfReports, map_minDimensions_userId_similarUsersWithScores,
                     map_opinionArgumentId_opinionArgument, map_topic_similarTopicsWithScores, number_of_examples, number_of_minDimensions, path_to_data_split):

    mapTrainAndDev_opinionArgumentId_userId, mapTest_opinionArgumentId_userId = map_opinion_ids_to_their_user_ids(data_split, map_opinionArgumentId_opinionArgument)
    map_minDimension_users = get_map_with_minDimensions_and_their_users(map_opinionArgumentId_opinionArgument)

    output = []

    for minDimension in number_of_minDimensions:

        print('-starting with dimension ' + str(minDimension))

        count = 0
        for opinionArgumentId in data_split.test:
            count += 1
            print('--next opinion id: ' + str(count) + '/' + str(len(data_split.test)))

            test_opinion_argument = map_opinionArgumentId_opinionArgument[opinionArgumentId]
            test_question_text = test_opinion_argument.question_text
            most_similar_topics_to_test_question_text = map_topic_similarTopicsWithScores[test_question_text]

            most_similar_topics_to_test_question_text_filtered = filter_out_similar_topics_that_are_not_in_the_train_or_dev_set(
                data_split, most_similar_topics_to_test_question_text
            )

            test_user_id = test_opinion_argument.user
            user_self_reports = map_userId_userSelfReports[test_user_id]
            most_similar_users_to_test_user = map_minDimensions_userId_similarUsersWithScores[minDimension][test_user_id]

            most_similar_users_to_test_user_filtered = filter_out_similar_users_that_are_not_in_the_train_or_dev_set_or_have_not_the_desired_minimum_dimension(
                mapTrainAndDev_opinionArgumentId_userId, map_minDimension_users, minDimension,
                most_similar_users_to_test_user
            )

            prompt_1_user = "Your task is, given a person's profile and a question, to predict the opinion of " \
                            "such a person regarding the question. You should reply with a stance (YES or NO) " \
                            "and a short single-sentence explanation explaining that opinion stance from the " \
                            "viewpoint of that person."

            prompt_2_user = 'Person A: [' + '\n' \
                            + 'political orientation: ' + str(user_self_reports.political_spectrum) + '\n' \
                            + 'relationship status: ' + str(user_self_reports.relationship) + '\n' \
                            + 'gender: ' + str(user_self_reports.gender) + '\n' \
                            + 'age: ' + str(user_self_reports.birthday) + '\n' \
                            + 'education level: ' + str(user_self_reports.education_level) + '\n' \
                            + 'ethnicity: ' + str(user_self_reports.ethnicity) + '\n' \
                            + 'income: ' + str(user_self_reports.income) + '\n' \
                            + 'working place: ' + str(user_self_reports.working_place) + '\n' \
                            + 'religious: ' + str(user_self_reports.religious) + ']' + '\n' \
                            + 'Question: "' + str(test_question_text) + '"'

            # find most similar (user,topic) pairs by product of rankings
            quadruple_userId_topicId_string_score = find_all_similar_user_topic_pairs_by_the_product_of_their_rankings(
                map_userId_userSelfReports, map_opinionArgumentId_opinionArgument, minDimension,
                most_similar_topics_to_test_question_text_filtered, most_similar_users_to_test_user_filtered,
                map_minDimension_users
            )

            keep_most_similar_distinct_user_topic_pairs(
                minDimension, number_of_examples, opinionArgumentId, output, path_to_data_split, prompt_1_user,
                prompt_2_user, quadruple_userId_topicId_string_score, test_user_id
            )

            # find most dissimilar users for most similar topic pairs
            negative_quadruple_userId_topicId_string_score = find_all_dissimilar_user_similar_topic_pairs_by_the_product_of_their_rankings(
                limit_of_similar_users_and_topics, map_userId_userSelfReports, map_opinionArgumentId_opinionArgument,
                minDimension, most_similar_topics_to_test_question_text_filtered, most_similar_users_to_test_user_filtered,
                map_minDimension_users
            )

            keep_most_dissimilar_distinct_user_topic_pairs(
                minDimension, number_of_examples, opinionArgumentId, output, path_to_data_split, prompt_1_user,
                prompt_2_user, negative_quadruple_userId_topicId_string_score, test_user_id
            )

    write_prompts_in_csv(limit_of_similar_users_and_topics, output, path_to_data_split)


def map_opinion_ids_to_their_user_ids(data_split, map_opinionArgumentId_opinionArgument):

    mapTrainAndDev_opinionArgumentId_userId = {}
    for opinionArgumentId in data_split.train:
        mapTrainAndDev_opinionArgumentId_userId[opinionArgumentId] = map_opinionArgumentId_opinionArgument[opinionArgumentId].user
    for opinionArgumentId in data_split.dev:
        mapTrainAndDev_opinionArgumentId_userId[opinionArgumentId] = map_opinionArgumentId_opinionArgument[opinionArgumentId].user

    mapTest_opinionArgumentId_userId = {}
    for opinionArgumentId in data_split.test:
        mapTest_opinionArgumentId_userId[opinionArgumentId] = map_opinionArgumentId_opinionArgument[opinionArgumentId].user

    return mapTrainAndDev_opinionArgumentId_userId, mapTest_opinionArgumentId_userId


def filter_out_similar_topics_that_are_not_in_the_train_or_dev_set(data_split, most_similar_topics_to_test_question_text):

    most_similar_topics_to_test_question_text_filtered = []
    for similar_topic in most_similar_topics_to_test_question_text:
        if similar_topic.id_target in data_split.train or similar_topic.id_target in data_split.dev:
            most_similar_topics_to_test_question_text_filtered.append(similar_topic)
    return most_similar_topics_to_test_question_text_filtered


def filter_out_similar_users_that_are_not_in_the_train_or_dev_set_or_have_not_the_desired_minimum_dimension(
        mapTrainAndDev_opinionArgumentId_userId, map_minDimension_users, minDimension, most_similar_users_to_test_user):

    most_similar_users_to_test_user_filtered = []
    for similar_user in most_similar_users_to_test_user:

        similar_user_j_id_has_min_dimensions = similar_user.user_j_id in map_minDimension_users[minDimension]

        similar_user_j_id_is_in_train_or_dev_set = False
        for opinionArgumentId in mapTrainAndDev_opinionArgumentId_userId:
            if mapTrainAndDev_opinionArgumentId_userId[opinionArgumentId] == similar_user.user_j_id:
                similar_user_j_id_is_in_train_or_dev_set = True
                break

        if similar_user_j_id_has_min_dimensions and similar_user_j_id_is_in_train_or_dev_set:
            most_similar_users_to_test_user_filtered.append(similar_user)

    return most_similar_users_to_test_user_filtered


def find_all_similar_user_topic_pairs_by_the_product_of_their_rankings(
        map_userId_userSelfReports, map_opinionArgumentId_opinionArgument, minDimension,
        most_similar_topics_to_test_question_text_filtered, most_similar_users_to_test_user_filtered,
        map_minDimension_users):

    quadruple_userId_topicId_string_score = []
    similar_user_count = 0
    for similar_user in most_similar_users_to_test_user_filtered:

        if similar_user.user_j_id not in map_minDimension_users[minDimension]:
            continue

        similar_user_self_reports = map_userId_userSelfReports[similar_user.user_j_id]

        for similar_topic in most_similar_topics_to_test_question_text_filtered:
            opinion_argument_of_similar_question = map_opinionArgumentId_opinionArgument[similar_topic.id_target]

            if opinion_argument_of_similar_question.user == similar_user.user_j_id:
                similar_user_count += 1

                prompt_3_user = ': [' + '\n' \
                                + 'political orientation: ' + str(similar_user_self_reports.political_spectrum) + '\n' \
                                + 'relationship status: ' + str(similar_user_self_reports.relationship) + '\n' \
                                + 'gender: ' + str(similar_user_self_reports.gender) + '\n' \
                                + 'age: ' + str(similar_user_self_reports.birthday) + '\n' \
                                + 'education level: ' + str(similar_user_self_reports.education_level) + '\n' \
                                + 'ethnicity: ' + str(similar_user_self_reports.ethnicity) + '\n' \
                                + 'income: ' + str(similar_user_self_reports.income) + '\n' \
                                + 'working place: ' + str(similar_user_self_reports.working_place) + '\n' \
                                + 'religious: ' + str(similar_user_self_reports.religious) + ']' + '\n' \
                                + 'answered to one of the most similar questions ' + '\n' \
                                + '"' + opinion_argument_of_similar_question.question_text + '\" with the stance \"' + opinion_argument_of_similar_question.stance + '" ' \
                                + 'and the conclusion/explanation "' + opinion_argument_of_similar_question.conclusion + '"' + '\n\n' + '----------------------' + '\n\n'

                quadruple_userId_topicId_string_score.append(
                    Quadruple_UserId_TopicId_Prompt_Score(
                        similar_user.user_j_id,
                        similar_topic.id_target,
                        prompt_3_user,
                        similar_user.rank * similar_topic.rank
                    )
                )
    return quadruple_userId_topicId_string_score


def keep_most_similar_distinct_user_topic_pairs(minDimension, number_of_examples, opinionArgumentId, output,
                                                path_to_data_split, prompt_1_user, prompt_2_user,
                                                quadruple_userId_topicId_string_score, test_user_id):

    quadruple_userId_topicId_string_score.sort(key=lambda x: x.score)
    for max_number_of_examples_in_this_run in number_of_examples:
        prompt_3plus_user = ''
        seen_user_ids = []
        seen_topic_ids = []
        found_user_topic_examples_in_this_run = 0
        for quadruple in quadruple_userId_topicId_string_score:
            if quadruple.user_id in seen_user_ids or quadruple.topic_id in seen_topic_ids:
                continue

            if found_user_topic_examples_in_this_run == max_number_of_examples_in_this_run:
                break
            else:
                found_user_topic_examples_in_this_run += 1

            prompt_3plus_user += 'Person ' + str(chr(ord('@') + found_user_topic_examples_in_this_run + 1)) + quadruple.prompt
            seen_user_ids.append(quadruple.user_id)
            seen_topic_ids.append(quadruple.topic_id)

        output.append([
            opinionArgumentId,
            path_to_data_split,
            minDimension,
            max_number_of_examples_in_this_run,
            test_user_id,
            'positive',
            seen_user_ids[:max_number_of_examples_in_this_run],
            seen_topic_ids[:max_number_of_examples_in_this_run],
            prompt_1_user,
            prompt_2_user,
            prompt_3plus_user
        ])


def find_all_dissimilar_user_similar_topic_pairs_by_the_product_of_their_rankings(
        limit_of_similar_users_and_topics, map_userId_userSelfReports, map_opinionArgumentId_opinionArgument,
        minDimension, most_similar_topics_to_test_question_text_filtered, most_similar_users_to_test_user_filtered,
        map_minDimension_users):

    quadruple_userId_topicId_string_score = []
    dissimilar_user_count = limit_of_similar_users_and_topics
    for dissimilar_user in most_similar_users_to_test_user_filtered:

        if dissimilar_user.user_j_id not in map_minDimension_users[minDimension]:
            continue

        dissimilar_user_self_reports = map_userId_userSelfReports[dissimilar_user.user_j_id]

        for similar_topic in most_similar_topics_to_test_question_text_filtered:
            opinion_argument_of_similar_question = map_opinionArgumentId_opinionArgument[similar_topic.id_target]

            if opinion_argument_of_similar_question.user == dissimilar_user.user_j_id:
                dissimilar_user_count += 1

                prompt_3_user = ': [' + '\n' \
                                + 'political orientation: ' + str(
                    dissimilar_user_self_reports.political_spectrum) + '\n' \
                                + 'relationship status: ' + str(
                    dissimilar_user_self_reports.relationship) + '\n' \
                                + 'gender: ' + str(dissimilar_user_self_reports.gender) + '\n' \
                                + 'age: ' + str(dissimilar_user_self_reports.birthday) + '\n' \
                                + 'education level: ' + str(
                    dissimilar_user_self_reports.education_level) + '\n' \
                                + 'ethnicity: ' + str(dissimilar_user_self_reports.ethnicity) + '\n' \
                                + 'income: ' + str(dissimilar_user_self_reports.income) + '\n' \
                                + 'working place: ' + str(
                    dissimilar_user_self_reports.working_place) + '\n' \
                                + 'religious: ' + str(dissimilar_user_self_reports.religious) + ']' + '\n' \
                                + 'answered to one of the most similar questions ' + '\n' \
                                + '"' + opinion_argument_of_similar_question.question_text + '\" with the stance \"' + opinion_argument_of_similar_question.stance + '" ' \
                                + 'and the conclusion/explanation "' + opinion_argument_of_similar_question.conclusion + '"' + '\n\n' + '----------------------' + '\n\n'

                quadruple_userId_topicId_string_score.append(
                    Quadruple_UserId_TopicId_Prompt_Score(
                        dissimilar_user.user_j_id,
                        similar_topic.id_target,
                        prompt_3_user,
                        dissimilar_user.negative_rank * similar_topic.rank
                    )
                )

    return quadruple_userId_topicId_string_score


def keep_most_dissimilar_distinct_user_topic_pairs(
        minDimension, number_of_examples, opinionArgumentId, output, path_to_data_split, prompt_1_user, prompt_2_user,
        quadruple_userId_topicId_string_score, test_user_id):

    quadruple_userId_topicId_string_score.sort(key=lambda x: x.score)
    for max_number_of_examples_in_this_run in number_of_examples:
        prompt_3plus_user = ''
        seen_user_ids = []
        seen_topic_ids = []
        found_user_topic_examples_in_this_run = 0
        for quadruple in quadruple_userId_topicId_string_score:
            if quadruple.user_id in seen_user_ids or quadruple.topic_id in seen_topic_ids:
                continue

            if found_user_topic_examples_in_this_run == max_number_of_examples_in_this_run:
                break
            else:
                found_user_topic_examples_in_this_run += 1

            prompt_3plus_user += 'Person ' + str(chr(ord('@') + found_user_topic_examples_in_this_run + max_number_of_examples_in_this_run + 1)) + quadruple.prompt
            seen_user_ids.append(quadruple.user_id)
            seen_topic_ids.append(quadruple.topic_id)

        output.append([
            opinionArgumentId,
            path_to_data_split,
            minDimension,
            max_number_of_examples_in_this_run,
            test_user_id,
            'negative',
            seen_user_ids[:max_number_of_examples_in_this_run],
            seen_topic_ids[:max_number_of_examples_in_this_run],
            prompt_1_user,
            prompt_2_user,
            prompt_3plus_user
        ])


def write_prompts_in_csv(limit_of_similar_users_and_topics, output, path_to_data_split):
    limit_of_similar_users_and_topics_path_output_string = '_withLimitOf' + str(limit_of_similar_users_and_topics)
    with open(os.path.join('data', 'prompts_for_chat_gpt_' + re.sub(r'\W+', '', path_to_data_split) + str(
            limit_of_similar_users_and_topics_path_output_string) + '.csv'), 'w', newline='\n',
              encoding='UTF8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL, delimiter=';')

        writer.writerow(
            ['opinion_argument_id',
             'path_to_data_split',
             'minDimension',
             'max_number_of_examples_in_this_run',
             'test_user_id',
             'positive_or_negative_example',
             'similar_user_ids',
             'similar_topic_ids',
             'prompt_1_user',
             'prompt_2_user',
             'prompt_3plus_user'
             ]
        )

        writer.writerows(output)

        file.flush()
        file.close()
