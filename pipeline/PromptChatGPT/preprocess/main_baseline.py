from pipeline.PromptChatGPT.preprocess.step_1_opinion_argument_parser import get_map_with_paths_to_train_dev_test_splits, generate_map_with_ids_to_opinion_arguments, test_mapping_of_train_dev_test_splits
from pipeline.PromptChatGPT.preprocess.step_2_most_similar_topic_searcher import get_the_n_most_similar_topics_for_each_topic
from pipeline.PromptChatGPT.preprocess.step_3_most_similar_users_searcher import min_dimensions, max_dimensions, get_the_n_most_similar_users_for_each_user
from pipeline.PromptChatGPT.preprocess.step_4_generate_prompts import generate_prompts


def main():

    map_path_trainDevTestSplits = get_map_with_paths_to_train_dev_test_splits()

    limit_of_similar_users_and_topics = 2000
    number_of_examples = [10]  # [0, 1, 3, 5, 10]
    number_of_minDimensions = range(min_dimensions, max_dimensions)

    for path_to_data_split in map_path_trainDevTestSplits:

        data_split = map_path_trainDevTestSplits[path_to_data_split]

        # argument_id --- link, question_text, category, tags, user, stance, argument_title, text, conclusion
        map_opinionArgumentId_opinionArgument = generate_map_with_ids_to_opinion_arguments(data_split)
        test_mapping_of_train_dev_test_splits(data_split, map_opinionArgumentId_opinionArgument)
        print("finished: " + str("generate_map_with_ids_to_opinion_arguments"))

        # topic_original --- id_original, topic_original, id_target, topic_target, cosine_similarity
        map_topic_similarTopicsWithScores = get_the_n_most_similar_topics_for_each_topic(map_opinionArgumentId_opinionArgument, 'all-MiniLM-L6-v2', 10*limit_of_similar_users_and_topics)
        print("finished: " + str("get_the_n_most_similar_topics_for_each_topic"))

        # minimumFilledDimensions --- userId --- user_i_id, user_j_id, euclidean_distance
        map_userId_userSelfReports, map_minimumFilledDimensions_userId_similarUsersWithScores = get_the_n_most_similar_users_for_each_user(map_opinionArgumentId_opinionArgument, limit_of_similar_users_and_topics)
        print("finished: " + str("get_the_n_most_similar_users_for_each_user"))

        # generate prompts
        generate_prompts(data_split, limit_of_similar_users_and_topics,
                         map_userId_userSelfReports, map_minimumFilledDimensions_userId_similarUsersWithScores,
                         map_opinionArgumentId_opinionArgument, map_topic_similarTopicsWithScores, number_of_examples, number_of_minDimensions,
                         path_to_data_split)

        print("finished: " + str("generate_prompts"))


if __name__ == '__main__':
    main()
