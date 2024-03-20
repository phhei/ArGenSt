from pipeline.data_gathering.users.UserSelfReport import UserSelfReport


def get_users_with_self_reports(connection, required_num_of_not_null_entries):
    cursor = connection.cursor()
    cursor.execute("SELECT "

                   "url AS url,"
                   "party AS political_spectrum,"
                   "relationship AS relationship,"
                   "gender AS gender,"
                   "birthday AS birthday,"
                   "education AS education_level,"
                   "ethnicity AS ethnicity,"
                   "income AS income,"
                   "occupation AS working_place,"
                   "religion AS religious,"
                   "number_of_all_debates AS number_of_debates,"
                   "last_online AS last_online "

                   "FROM users"
                   )
    rows = cursor.fetchall()

    map_user_self_report = {}
    for row in rows:
        url = row[0]
        political_spectrum = row[1]
        relationship = row[2]
        gender = row[3]
        age = row[4]
        education_level = row[5]
        ethnicity = row[6]
        income = row[7]
        working_place = row[8]
        religious = row[9]
        number_of_debates = row[10]
        last_online = row[11]

        if url not in map_user_self_report:
            self_report = UserSelfReport(url, political_spectrum, relationship, gender, age, education_level,
                                         ethnicity, income, working_place, religious, number_of_debates, last_online)

            if self_report.count_number_of_not_null_entries() >= required_num_of_not_null_entries:
                map_user_self_report[url] = self_report

    return map_user_self_report
