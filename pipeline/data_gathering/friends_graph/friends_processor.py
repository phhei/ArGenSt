def get_friends_rows(connection):
    cursor = connection.cursor()
    cursor.execute("SELECT "
                   "user, friend "
                   "FROM friends"
                   )
    rows = cursor.fetchall()
    return rows
