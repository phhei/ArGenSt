import csv
import os
import random

path_to_extern_opinion_arguments_file = os.path.join('..', '..', 'data', 'TITLEA~1.CSV')
path_to_extern_opinion_arguments_with_real_names_file = os.path.join('..', '..', 'data', 'extern_opinion_arguments_with_real_names.csv')


def process_opinion_arguments(connection):
    rows_from_extern_opinion_arguments = get_rows_from_extern_opinion_arguments(path_to_extern_opinion_arguments_file)
    map_argumentId_userId = create_map_with_argumentId_to_userId(connection)
    output = update_rows_with_real_names(rows_from_extern_opinion_arguments, map_argumentId_userId)
    write_opinion_arguments_in_csv(output, True)


def get_rows_from_extern_opinion_arguments (path_to_extern_opinion_arguments_file):
    output = []

    with open(path_to_extern_opinion_arguments_file, newline='\n', encoding="utf8") as csv_file_read:
        csv_reader = csv.reader(csv_file_read, delimiter=',', quotechar='"')
        next(csv_reader, None)  # skip the headers
        for row in csv_reader:
            # argument_id,link,question_text,category,tags,user,stance,argument_title,text,conclusion
            output.append(row)

    return output


def create_map_with_argumentId_to_userId(connection):
    cursor = connection.cursor()
    cursor.execute("SELECT "
                   "* "
                   "FROM detailed_opinion_arguments"
                   )
    rows = cursor.fetchall()

    map_argumentId_userId = {}
    for row in rows:
        map_argumentId_userId[row[0]] = row[1]

    return map_argumentId_userId


def update_rows_with_real_names(rows_from_extern_opinion_arguments, map_argumentId_userId):
    output = []
    for row in rows_from_extern_opinion_arguments:
        # argument_id, link, question_text, category, tags, user, stance, argument_title, text, conclusion
        output.append(
            [
                row[0],  # argument_id
                row[1],  # link
                row[2],  # question_text
                row[3],  # category
                row[4],  # tags
                map_argumentId_userId.get(row[0]),  # user
                row[6],  # stance
                row[7],  # argument_title
                row[8],  # text
                row[9],  # conclusion
            ]
        )
    return output


def write_opinion_arguments_in_csv(output, shuffle):
    with open(path_to_extern_opinion_arguments_with_real_names_file, 'w', newline='\n', encoding='UTF8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL, delimiter=';')

        writer.writerow(['argument_id', 'link', 'question_text', 'category', 'tags', 'user', 'stance', 'argument_title', 'text', 'conclusion'])

        if shuffle:
            random.shuffle(output)

        writer.writerows(output)

        file.flush()
        file.close()
