import csv
import os
import re

from scipy.spatial import distance

from pipeline.PromptChatGPT.preprocess.SimilarUserWithScore import SimilarUserWithScore
from pipeline.data_gathering.users.UserSelfReport import UserSelfReport

min_dimensions = 0
max_dimensions = 13


def get_the_n_most_similar_users_for_each_user(map_opinionArgumentId_opinionArgument, limit_of_similar_users_and_topics):

    map_minDimension_users = get_map_with_minDimensions_and_their_users(map_opinionArgumentId_opinionArgument)
    map_userId_userSelfReports = parse_user_self_reports(map_opinionArgumentId_opinionArgument)
    map_tower_value_dimensions = map_values_of_each_towers_attributes_to_final_dimensions()
    map_userId_vector = map_user_ids_to_vectors_of_generalized_dimensions(map_userId_userSelfReports, map_tower_value_dimensions)
    map_minimumFilledDimensions_userId_similarUsersWithScores = find_most_similar_users(map_minDimension_users, map_userId_vector, limit_of_similar_users_and_topics)

    return map_userId_userSelfReports, map_minimumFilledDimensions_userId_similarUsersWithScores


def get_map_with_minDimensions_and_their_users(map_opinionArgumentId_opinionArgument):

    # get user ids of train dev test data to proceed with these only
    user_ids_in_train_dev_test_data = set()
    for opinionArgumentId in map_opinionArgumentId_opinionArgument:
        user_ids_in_train_dev_test_data.add(map_opinionArgumentId_opinionArgument[opinionArgumentId].user)


    map_minDimensions_userId = {}
    for minDimensions in range(min_dimensions, max_dimensions):

        map_minDimensions_userId[minDimensions] = []

        with open(os.path.join('data', 'user_with_num_of_not_null_entries_' + str(minDimensions) + '.csv'), newline='\n', encoding="utf8") as csv_file_read:
            csv_reader = csv.reader(csv_file_read, delimiter=';', quotechar='"')
            next(csv_reader, None)  # skip the headers

            # "user_id";"political_spectrum";"relationship";"gender";"birthday";"current_age";"age_of_last_login";"education_level";"ethnicity";"income";"working_place";"religious";"number_of_debates"
            for row in csv_reader:

                if row[0] not in user_ids_in_train_dev_test_data:
                    continue

                map_minDimensions_userId[minDimensions].append(row[0])

    return map_minDimensions_userId


def parse_user_self_reports(map_opinionArgumentId_opinionArgument):

    # get user ids of train dev test data to proceed with these only
    user_ids_in_train_dev_test_data = set()
    for opinionArgumentId in map_opinionArgumentId_opinionArgument:
        user_ids_in_train_dev_test_data.add(map_opinionArgumentId_opinionArgument[opinionArgumentId].user)

    map_userId_userSelfReports = {}

    with open(os.path.join('data', 'user_with_num_of_not_null_entries_0.csv'), newline='\n', encoding="utf8") as csv_file_read:
        csv_reader = csv.reader(csv_file_read, delimiter=';', quotechar='"')
        next(csv_reader, None)  # skip the headers

        # "user_id";"political_spectrum";"relationship";"gender";"birthday";"current_age";"age_of_last_login";"education_level";"ethnicity";"income";"working_place";"religious";"number_of_debates"
        for row in csv_reader:

            if row[0] not in user_ids_in_train_dev_test_data:
                continue

            # url, political_spectrum, relationship, gender, birthday, education_level, ethnicity, income, working_place, religious, number_of_debates, last_online
            map_userId_userSelfReports[row[0]] = UserSelfReport(
                row[0],  # user_id -> url
                row[1],  # political_spectrum -> political_spectrum
                row[2],  # relationship -> relationship
                row[3],  # gender -> gender
                row[5],  # birthday -> current_age
                row[7],  # education_level -> education_level
                row[8],  # ethnicity -> ethnicity
                row[9],  # income -> income
                row[10],  # working_place -> working_place
                row[11],  # religious -> religious
                row[12],  # number_of_debates -> number_of_debates
                row[6]  # age_of_last_login -> last_online
            )

    return map_userId_userSelfReports


def map_values_of_each_towers_attributes_to_final_dimensions():

    # get paths to towers
    linear_user_profile_encoder_embeddings = []
    for (dir_path, dir_names, file_names) in os.walk(os.path.join('..', '..', 'pipeline', 'StanceClassifier', 'submodules', 'towers', 'LinearUserProfileEncoderEmbeddings')):
        for file_name in file_names:
            linear_user_profile_encoder_embeddings.append(os.path.join(str(dir_path), str(file_name)))

    # map values of each towers' attributes to final dimensions
    map_tower_value_dimensions = {}
    for path_to_tower in linear_user_profile_encoder_embeddings:
        # convert file name to tower name
        match = re.search(r'.*/([a-zA-Z0-9_]+)\.tsv', path_to_tower)
        if match:
            tower = match.group(1)
        else:
            raise Exception('could not find tower: ' + str(path_to_tower))

        map_tower_value_dimensions[tower] = {}
        with open(path_to_tower) as tst_file:
            rows = csv.reader(tst_file, delimiter="\t", quotechar='"')
            for row in rows:
                if row[0].startswith('#'):
                    continue
                map_tower_value_dimensions[tower][row[0]] = [float(i) for i in row[1:]]

    return map_tower_value_dimensions


def map_user_ids_to_vectors_of_generalized_dimensions(map_userId_userSelfReports, map_tower_value_dimensions):

    map_userId_vector = {}

    for user_id in map_userId_userSelfReports:

        if user_id not in map_userId_vector:
            map_userId_vector[user_id] = []

        user_self_report = map_userId_userSelfReports[user_id]

        list_entityName_entityValue = [
            ('age', user_self_report.birthday),
            ('education_level', user_self_report.education_level),
            ('ethnicity', user_self_report.ethnicity),
            ('gender', user_self_report.gender),
            ('income', user_self_report.income),
            ('political_spectrum', user_self_report.political_spectrum),
            ('relationship', user_self_report.relationship),
            ('religious', user_self_report.religious),
            ('working_place', user_self_report.working_place),
        ]

        for pair in list_entityName_entityValue:
            entity_name = pair[0]
            entity_value = pair[1]

            try:
                map_userId_vector[user_id].extend(
                    map_tower_value_dimensions[entity_name][entity_value]
                )
            except:
                map_userId_vector[user_id].extend(
                    map_tower_value_dimensions[entity_name]['default']
                )

    return map_userId_vector


def find_most_similar_users(map_minDimension_users, map_userId_vector, limit_of_similar_users_and_topics):

    map_minDimensions_userId_similarUsersWithScores = {}

    for minDimensions in map_minDimension_users:

        if minDimensions not in map_minDimensions_userId_similarUsersWithScores:
            map_minDimensions_userId_similarUsersWithScores[minDimensions] = {}

        print('find most similar users for dimension ' + str(minDimensions))

        count = 0
        for user_i_id in map_userId_vector:
            count += 1
            print('progress for users: ' + str(count) + '/' + str(len(map_userId_vector)))

            user_i_vector = map_userId_vector[user_i_id]

            tmp_list = []

            for user_j_id in map_userId_vector:

                if user_j_id not in map_minDimension_users[minDimensions]:
                    continue

                user_j_vector = map_userId_vector[user_j_id]

                euclidean_distance = distance.euclidean(user_i_vector, user_j_vector)

                tmp_list.append(
                    SimilarUserWithScore(
                        user_i_id, user_j_id, euclidean_distance
                    )
                )


            # assign ranking for most similar users
            tmp_list.sort(key=lambda x: x.euclidean_distance)
            current_rank = 1
            last_euclidean_distance = tmp_list[0].euclidean_distance
            for similar_user in tmp_list:
                if similar_user.euclidean_distance > last_euclidean_distance:
                    current_rank += 1
                    last_euclidean_distance = similar_user.euclidean_distance
                similar_user.rank = current_rank

            # assign ranking for most dissimilar users
            negative_tmp_list = tmp_list
            negative_tmp_list.sort(key=lambda x: x.euclidean_distance, reverse=True)
            negative_current_rank = 1
            negative_last_euclidean_distance = negative_tmp_list[0].euclidean_distance
            for dissimilar_user in negative_tmp_list:
                if dissimilar_user.euclidean_distance < negative_last_euclidean_distance:
                    negative_current_rank += 1
                    negative_last_euclidean_distance = dissimilar_user.euclidean_distance
                dissimilar_user.negative_rank = negative_current_rank

            # make final list of similar and dissimilar users
            final_tmp_list = []
            final_tmp_list_ids = set()


            for similar_user in (tmp_list[:min(limit_of_similar_users_and_topics, len(tmp_list))] + negative_tmp_list[:min(limit_of_similar_users_and_topics, len(tmp_list))]):
                if similar_user.user_j_id in final_tmp_list_ids:
                    continue
                else:
                    final_tmp_list.append(similar_user)
                    final_tmp_list_ids.add(similar_user.user_j_id)

            map_minDimensions_userId_similarUsersWithScores[minDimensions][user_i_id] = final_tmp_list

    return map_minDimensions_userId_similarUsersWithScores
