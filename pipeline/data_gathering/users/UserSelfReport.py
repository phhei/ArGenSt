class UserSelfReport:

    def __init__(self,
                 url, political_spectrum, relationship, gender, birthday, education_level, ethnicity, income,
                 working_place, religious, number_of_debates, last_online):
        self.url = url
        self.political_spectrum = political_spectrum
        self.relationship = relationship
        self.gender = gender
        self.birthday = birthday
        self.education_level = education_level
        self.ethnicity = ethnicity
        self.income = income
        self.working_place = working_place
        self.religious = religious
        self.number_of_debates = number_of_debates
        self.last_online = last_online

    def count_number_of_not_null_entries(self):

        self_report_entities = [self.political_spectrum, self.relationship, self.gender, self.birthday,
                                self.education_level, self.ethnicity, self.income, self.working_place, self.religious]

        num_of_not_null_entries = 0
        for self_report_entity in self_report_entities:

            if len(str(self_report_entity)) != 0 and not str(self_report_entity) == '- Private -' and not str(self_report_entity) == 'Not Saying' and not str(self_report_entity) == 'Prefer not to say':
                num_of_not_null_entries += 1

        return num_of_not_null_entries


    def __str__(self):
        return "{}: {}".format(
            self.url,
            " / ".join([f"{prop}: {value}" for prop, value in self.__dict__.items()])
        )

    def __hash__(self):
        return hash(self.url)

    def __eq__(self, other):
        return isinstance(other, UserSelfReport) and self.url == other.url
