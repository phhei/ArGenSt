import csv
import os.path
import re
import sqlite3
from sqlite3 import Error

from pipeline.data_gathering.friends_graph.friends_processor import get_friends_rows
from pipeline.data_gathering.opinion_arguments.Poll_Vote import PollVote
from pipeline.data_gathering.opinion_arguments.extern_opinion_argument_processor import process_opinion_arguments
from pipeline.data_gathering.tower_generator.AnnotationParserAndTowerGenerator import parse_annotations_and_write_towers
from pipeline.data_gathering.users.userProcessor import get_users_with_self_reports

path_to_database = os.path.abspath(os.path.join('..', '..', 'data', 'debateOrg', 'ddo-V2.db'))

random_state = 42


def gather_all_data():
    connection = connect_to_SQLite(path_to_database)
    with connection:
        make_datasets(connection)


def connect_to_SQLite(db_file):
    connection = None
    try:
        connection = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return connection


def make_datasets(connection):

    # 1 table with all topics (votes, opinions) (1 topic per line)
    process_opinion_arguments(connection)


    # 1 table showing the friendship network of the users
    friends_rows = get_friends_rows(connection)
    all_users_with_poll_votes = get_all_users_with_poll_votes(connection)
    write_csv_file_with_friends_graph(friends_rows, all_users_with_poll_votes)


    # 1 table for all usres (all properties of a user in one line)
    for i in range(0, 10):
        map_user_self_report = get_users_with_self_reports(connection, i)
        write_csv_file_with_users(map_user_self_report, i)


def write_csv_file_with_friends_graph(friends_rows, all_users_with_poll_votes):
    rows_friends_graph = []
    header_friends_graph = ['user_id', 'friend_user_id']
    rows_friends_graph.append(header_friends_graph)

    all_users_in_friends_rows = set()

    with open(os.path.join('..', '..', 'data', 'friends_graph.csv'), 'w', newline='\n', encoding='UTF8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL, delimiter=';')

        for row in friends_rows:
            user = row[0]
            friend = row[1]

            all_users_in_friends_rows.add(user)
            all_users_in_friends_rows.add(friend)

            friend_relation = [user, friend]
            rows_friends_graph.append(friend_relation)

        for user in all_users_with_poll_votes:
            if user not in all_users_in_friends_rows:
                friend_relation = [user, None]
                rows_friends_graph.append(friend_relation)
                print(friend_relation)

        writer.writerows(rows_friends_graph)

        file.flush()
        file.close()


def write_csv_file_with_users(map_user_selfReport, num_of_not_null_entries):

    header_user = ['user_id',
                   'political_spectrum',
                   'relationship',
                   'gender',
                   'birthday', 'current_age', 'age_of_last_login',
                   'education_level',
                   'ethnicity',
                   'income',
                   'working_place',
                   'religious',
                   'number_of_debates']

    rows_user = [header_user]
    for user in map_user_selfReport:
        sr = map_user_selfReport[user]
        self_reports = [sr.url,
                        sr.political_spectrum,
                        sr.relationship,
                        sr.gender,
                        sr.birthday, map_birthday_to_age(sr.birthday), map_birthday_to_age_of_last_login(sr.birthday, sr.last_online),
                        sr.education_level,
                        sr.ethnicity,
                        sr.income,
                        sr.working_place,
                        sr.religious,
                        sr.number_of_debates]

        rows_user.append(self_reports)

    with open(os.path.join('..', '..', 'data', 'user_with_num_of_not_null_entries_' + str(num_of_not_null_entries) + '.csv'), 'w', newline='\n', encoding='UTF8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL, delimiter=';')
        writer.writerows(rows_user)

        file.flush()
        file.close()


def map_birthday_to_age(birthday):
    try:
        result = re.search(".* (\d+)", birthday)

        age = 2023 - int(result.group(1))

        return age
    except:
        return 'Not Saying'


def map_birthday_to_age_of_last_login(birthday, last_online):
    try:
        age_regex = re.search(".* (\d+)", birthday)
        year_of_birth = int(age_regex.group(1))

        last_online_regex = re.search("(\d+) (Day|Week|Month|Year)", last_online)
        time_unit = last_online_regex.group(2)
        time_value = int(last_online_regex.group(1))
        if time_unit == 'Year':
            last_online_age = 2023 - time_value - year_of_birth
        else:
            last_online_age = 2023 - year_of_birth

        return last_online_age
    except:
        return 'Not Saying'


def get_all_users_with_poll_votes(connection):
    rows = get_rows_from_SQLite(connection)
    poll_votes = extract_poll_votes_from_rows(rows)

    all_users_with_poll_votes = set()
    for poll_vote in poll_votes:
        all_users_with_poll_votes.add(poll_vote.user)

    return all_users_with_poll_votes


def get_rows_from_SQLite(connection):
    cursor = connection.cursor()
    cursor.execute("SELECT "

                   "user,"
                   "link,"
                   "title,"
                   "vote,"
                   "explanation "

                   "FROM poll_votes"
                   )
    rows = cursor.fetchall()

    return rows


def extract_poll_votes_from_rows(rows):
    poll_votes = []

    for row in rows:
        user = row[0]
        link = row[1]
        title = row[2]
        vote = row[3]
        explanation = row[4]

        poll_votes.append(PollVote(user, link, title, vote, explanation))

    return poll_votes


if __name__ == '__main__':
    parse_annotations_and_write_towers()
    gather_all_data()
