import csv
import os.path

path_to_friends_graph_csv = os.path.join('..', '..', '..', 'data', 'friends_graph.csv')
path_to_friends_graph_json = os.path.join('..', '..', '..', 'data', 'friends_graph.json')


def main(friendship_strength: float):
    with open(path_to_friends_graph_csv, newline='\n') as csv_file_friends_graph:
        friends_graph_csv_rows = csv.reader(csv_file_friends_graph, delimiter=';', quotechar='"')  # "user_id";"friend_user_id"

        user_ids = set()
        user_id_friendship_relations = []
        for row in list(friends_graph_csv_rows)[1:]:  # user_id_1, user_id_2
            user_1 = row[0]
            user_2 = row[1]
            user_ids.add(user_1)
            if len(user_2) > 0:
                user_ids.add(user_2)
                user_id_friendship_relations.append((user_1, user_2, friendship_strength))

        output = "\"user_ids\": [{}], \n".format(", ".join(map(lambda u: "\"{}\"".format(u), user_ids)))
        output += "\"user_id_friendship_relations\": [{}]".format(
            ", ".join(map(lambda u: "[\"{}\", \"{}\", {}]".format(*u), user_id_friendship_relations))
        )

        with open(path_to_friends_graph_json, 'w', newline='\n') as json_file_friends_graph:
            json_file_friends_graph.write(output)
            json_file_friends_graph.flush()
            json_file_friends_graph.close()


if __name__ == "__main__":
    main(friendship_strength=.5)
