from sentence_transformers import SentenceTransformer, util

from pipeline.PromptChatGPT.preprocess.SimilarTopicWithScore import SimilarTopicWithScore


def get_the_n_most_similar_topics_for_each_topic(map_id_opinionArgument, model_name, limit_of_similar_users_and_topics):

    opinion_argument_ids, question_texts = get_ids_and_question_texts(map_id_opinionArgument)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(question_texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings, embeddings)

    print('progress for finding similar topics:')

    map_sentence_similarSentencesWithScores = {}
    for i in range(len(question_texts)):
        map_sentence_similarSentencesWithScores[question_texts[i]] = []
        tmp_list = []

        for j in range(len(question_texts)):

            tmp_list.append(
                SimilarTopicWithScore(
                    opinion_argument_ids[i],
                    question_texts[i],
                    opinion_argument_ids[j],
                    question_texts[j],
                    cosine_scores[i][j]
                )
            )

        tmp_list.sort(key=lambda x: x.cosine_similarity, reverse=True)

        # assign ranking
        current_rank = 1
        last_cosine_similarity = tmp_list[0].cosine_similarity
        for similarTopic in tmp_list:
            if similarTopic.cosine_similarity < last_cosine_similarity:
                current_rank += 1
                last_cosine_similarity = similarTopic.cosine_similarity
            similarTopic.rank = current_rank

        map_sentence_similarSentencesWithScores[question_texts[i]] = tmp_list[:min(limit_of_similar_users_and_topics, len(tmp_list))]

        print(str(i) + '/' + str(len(question_texts)))

    return map_sentence_similarSentencesWithScores


def get_ids_and_question_texts(map_id_opinionArgument):
    ids = []
    question_texts = []
    for opinionArgumentId in map_id_opinionArgument:
        opinion_argument = map_id_opinionArgument[opinionArgumentId]
        question_texts.append(opinion_argument.question_text)
        ids.append(opinionArgumentId)
    return ids, question_texts
