import json
from idlelib.tooltip import Hovertip
from tkinter import *
from tkinter import messagebox as mb


COMMENT_DEFAULT_STRING = "comment..."


def display_title():
    title = Label(gui, text="SOCIAL - Stakeholder Opinion prediCtIon Annotation tooL", width=50, bg="yellow", fg="black", font=("ariel", 20, "bold"))
    title.place(x=0, y=2)


class Annotation:
    def __init__(self):
        self.q_number = 0
        display_title()
        self.display_question()

        self.opt_selected_does_stance_make_sense = IntVar()

        self.opt_selected_why_does_stance_not_make_sense__stance_is_wrong = IntVar()
        self.opt_selected_why_does_stance_not_make_sense__generated_text_is_not_understandable = IntVar()
        self.opt_selected_why_does_stance_not_make_sense__generated_text_other_than_topic = IntVar()

        self.opt_selected_generated_text_contains_all_from_original = IntVar()

        self.opt_selected_generated_text_contains_realistic_additional_elements = IntVar()

        self.opts = self.display_answers()
        self.display_options()
        self.buttons()
        self.data_size = len(questions)

    def next_btn(self):

        # check whether next example can be shown
        if self.opt_selected_does_stance_make_sense.get() == 0:
            return

        if self.opt_selected_does_stance_make_sense.get() == 1 or self.opt_selected_does_stance_make_sense.get() == 2:
            if self.opt_selected_why_does_stance_not_make_sense__stance_is_wrong.get() == 0 \
                    and self.opt_selected_why_does_stance_not_make_sense__generated_text_is_not_understandable.get() == 0 \
                    and self.opt_selected_why_does_stance_not_make_sense__generated_text_other_than_topic.get() == 0:
                return

        if self.opt_selected_does_stance_make_sense.get() == 3 or self.opt_selected_does_stance_make_sense.get() == 4:
            if self.opt_selected_generated_text_contains_all_from_original.get() == 0 \
                    or self.opt_selected_generated_text_contains_realistic_additional_elements.get() == 0:
                return


        # save annotation with not blanked out fields and proceed with next
        for entry in self.stance_makes_no_sense_because_q_list:
            entry.configure(state=NORMAL)
        for entry in self.generated_text_contains_everything_from_ground_truth_q_list:
            entry.configure(state=NORMAL)
        for entry in self.generated_text_contains_additional_realistic_elements_q_list:
            entry.configure(state=NORMAL)

        if self.opt_selected_does_stance_make_sense.get() == 1 or self.opt_selected_does_stance_make_sense.get() == 2:
            answers.append(
                ((self.opt_selected_does_stance_make_sense.get(),
                  self.comment_does_stance_make_sense.get("1.0", "end-1c")),
                 (self.opt_selected_why_does_stance_not_make_sense__stance_is_wrong.get(),
                  self.opt_selected_why_does_stance_not_make_sense__generated_text_is_not_understandable.get(),
                  self.opt_selected_why_does_stance_not_make_sense__generated_text_other_than_topic.get(),
                  self.comment_why_does_stance_not_make_sense__comment.get("1.0", "end-1c")),
                 ("",
                  ""),
                 ("",
                  "")))

        elif self.opt_selected_does_stance_make_sense.get() == 3 or self.opt_selected_does_stance_make_sense.get() == 4:
            answers.append(
                ((self.opt_selected_does_stance_make_sense.get(),
                  self.comment_does_stance_make_sense.get("1.0", "end-1c")),
                 ("",
                  "",
                  "",
                  ""),
                 (self.opt_selected_generated_text_contains_all_from_original.get(),
                  self.comment_generated_text_contains_all_from_original.get("1.0", "end-1c")),
                 (self.opt_selected_generated_text_contains_realistic_additional_elements.get(),
                  self.comment_generated_text_contains_additional_realistic_elements.get("1.0", "end-1c"))))


        self.q_number += 1
        if self.q_number == self.data_size:
            mb.showinfo("Info", "Your annotations were saved!")
            gui.destroy()
        else:
            self.display_question()
            self.display_options()

    def back_btn(self):
        if self.q_number > 0:
            self.q_number -= 1
            answers.pop()
            self.display_question()
            self.display_options()

    def buttons(self):
        next_button = Button(gui, text="Next", command=self.next_btn, width=10, bg="blue", fg="white", font=("courier new", 10))
        next_button.place(x=150, y=970)
        back_button = Button(gui, text="Back", command=self.back_btn, width=10, bg="blue", fg="white", font=("courier new", 10))
        back_button.place(x=50, y=970)
        quit_button = Button(gui, text="Quit (this question will not be saved)", command=gui.destroy, width=50, bg="black", fg="white", font=("courier new", 10))
        quit_button.place(x=300, y=970)

    def display_options(self):
        val = 0
        self.opt_selected_does_stance_make_sense.set(0)
        self.opt_selected_why_does_stance_not_make_sense__stance_is_wrong.set(0)
        self.opt_selected_why_does_stance_not_make_sense__generated_text_is_not_understandable.set(0)
        self.opt_selected_why_does_stance_not_make_sense__generated_text_other_than_topic.set(0)
        self.opt_selected_generated_text_contains_all_from_original.set(0)
        self.opt_selected_generated_text_contains_realistic_additional_elements.set(0)

        self.comment_does_stance_make_sense.delete(1.0, END)
        self.comment_does_stance_make_sense.insert(END, COMMENT_DEFAULT_STRING)

        self.comment_why_does_stance_not_make_sense__comment.delete(1.0, END)
        self.comment_why_does_stance_not_make_sense__comment.insert(END, COMMENT_DEFAULT_STRING)

        self.comment_generated_text_contains_all_from_original.delete(1.0, END)
        self.comment_generated_text_contains_all_from_original.insert(END, COMMENT_DEFAULT_STRING)

        self.comment_generated_text_contains_additional_realistic_elements.delete(1.0, END)
        self.comment_generated_text_contains_additional_realistic_elements.insert(END, COMMENT_DEFAULT_STRING)


        for option in range(1, 5):
            self.opts[val]['text'] = option
            val += 1

    def display_question(self):

        x_pos = 10
        x_diff = 330

        y_pos = 50
        y_diff = 30

        # debate title
        label_debate_title = Label(gui, text="suppose the debate title is as follows:", width=60, font=("Times New Roman", 12, "bold"), anchor='w')
        label_debate_title.place(x=x_pos, y=y_pos)

        textbox_debate_title = Text(gui, height=1, width=60, bg="light yellow")
        textbox_debate_title.insert(END, str(questions[self.q_number][1]))
        textbox_debate_title.place(x=x_pos+x_diff, y=y_pos)
        textbox_debate_title.configure(state=DISABLED)

        # stakeholder group
        label_stakeholder_group = Label(gui, text="and the user's properties are:", width=60, font=("Times New Roman", 12, "bold"), anchor='w')
        label_stakeholder_group.place(x=x_pos, y=y_pos+y_diff)

        textbox_stakeholder_group = Text(gui, height=10, width=60, bg="light yellow")
        textbox_stakeholder_group.insert(END, str(questions[self.q_number][3]))
        textbox_stakeholder_group.place(x=x_pos+x_diff, y=y_pos+y_diff)
        textbox_stakeholder_group.configure(state=DISABLED)

        # vote
        label_vote_and_explanation = Label(gui, text="Further, the user's vote is:", width=60, font=("Times New Roman", 12, "bold"), anchor='w')
        label_vote_and_explanation.place(x=x_pos, y=y_pos+7*y_diff)

        textbox_vote_and_explanation = Text(gui, height=1, width=60, bg="light yellow")
        predicted_stance_str = 'yes' if ((questions[self.q_number][8] == 'fineTunedApproach' and float(questions[self.q_number][4]) >= 0.4890013039112091) or float(questions[self.q_number][4]) >= 0.5) else 'no'  # predicted stance
        textbox_vote_and_explanation.insert(END, predicted_stance_str)
        textbox_vote_and_explanation.place(x=x_pos+x_diff, y=y_pos+7*y_diff)
        textbox_vote_and_explanation.configure(state=DISABLED)

        # ground truth and generated explanations
        label_ground_truth_and_generated_explanation = Label(gui, text="Now compare the two explanations:", width=70, font=("Times New Roman", 12, "bold"), anchor='w')
        label_ground_truth_and_generated_explanation.place(x=x_pos, y=y_pos+9*y_diff)

        textbox_ground_truth_explanation = Text(gui, height=10, width=50, bg="light yellow")
        textbox_ground_truth_explanation.insert(END, "(original) explanation of the stance:" + "\n\n" + str(questions[self.q_number][6]))
        textbox_ground_truth_explanation.place(x=x_pos, y=y_pos+10*y_diff)
        textbox_ground_truth_explanation.configure(state=DISABLED)

        textbox_generated_explanation = Text(gui, height=10, width=50, bg="light yellow")
        textbox_generated_explanation.insert(END, "(generated) alternative explanation of the stance:" + "\n\n" + str(questions[self.q_number][7]))
        textbox_generated_explanation.place(x=x_pos+x_diff+80, y=y_pos+10*y_diff)
        textbox_generated_explanation.configure(state=DISABLED)


    def display_answers(self):

        x_pos = 10
        x_diff = 100

        y_diff = 30
        y_pos = 50 + 11*y_diff

        # todos and explanation
        label_todo_for_radio_buttons = Label(gui, text="Please rate the following questions (1 := no, 2 := rather no, 3 := rather yes, 4 := yes):", width=70, font=("Times New Roman", 12, "bold"), anchor='w')
        label_todo_for_radio_buttons.place(x=x_pos, y=y_pos+5*y_diff)


        # stance
        label_does_stance_make_sense = Label(gui, text="Does the stance make sense?", width=70, font=("Times New Roman", 12, "bold"), anchor='w')
        label_does_stance_make_sense.place(x=x_pos, y=y_pos+7*y_diff)

        self.stance_q_list = []
        tmp_x_pos = x_pos + 3*x_diff

        # add 4 radio buttons to choose between 1 (no), 2 (rather no), 3 (rather yes), and 4 (yes).
        # 1
        radio_does_stance_make_sense = Radiobutton(gui, text=1, variable=self.opt_selected_does_stance_make_sense,
                                                   value=1, font=("courier new", 10),
                                                   command=lambda: self.change_visibility_of_other_options(1))
        self.stance_q_list.append(radio_does_stance_make_sense)
        radio_does_stance_make_sense.place(x=tmp_x_pos, y=y_pos + 7 * y_diff)
        tmp_x_pos += 40

        # 2
        radio_does_stance_make_sense = Radiobutton(gui, text=2, variable=self.opt_selected_does_stance_make_sense,
                                                   value=2, font=("courier new", 10),
                                                   command=lambda: self.change_visibility_of_other_options(2))
        self.stance_q_list.append(radio_does_stance_make_sense)
        radio_does_stance_make_sense.place(x=tmp_x_pos, y=y_pos + 7 * y_diff)
        tmp_x_pos += 40

        # 3
        radio_does_stance_make_sense = Radiobutton(gui, text=3, variable=self.opt_selected_does_stance_make_sense,
                                                   value=3, font=("courier new", 10),
                                                   command=lambda: self.change_visibility_of_other_options(3))
        self.stance_q_list.append(radio_does_stance_make_sense)
        radio_does_stance_make_sense.place(x=tmp_x_pos, y=y_pos + 7 * y_diff)
        tmp_x_pos += 40

        # 4
        radio_does_stance_make_sense = Radiobutton(gui, text=4, variable=self.opt_selected_does_stance_make_sense,
                                                   value=4, font=("courier new", 10),
                                                   command=lambda: self.change_visibility_of_other_options(4))
        self.stance_q_list.append(radio_does_stance_make_sense)
        radio_does_stance_make_sense.place(x=tmp_x_pos, y=y_pos + 7 * y_diff)
        tmp_x_pos += 40


        self.comment_does_stance_make_sense = Text(gui, height=1, width=40, bg="light green")
        self.comment_does_stance_make_sense.insert(END, COMMENT_DEFAULT_STRING)
        self.comment_does_stance_make_sense.place(x=tmp_x_pos + 10, y=y_pos + 7 * y_diff + 10)
        self.stance_q_list.append(self.comment_does_stance_make_sense)

        # sub questions depending on stance
        self.display_if_stance_does_not_make_sense_explain_why(x_pos, x_diff, y_pos, y_diff)

        # if stance does make sense
        self.display_alternative_text_contains_everything_and_realistic_elements(x_pos, x_diff, y_pos, y_diff)

        for entry in self.stance_makes_no_sense_because_q_list:
            entry.configure(state=DISABLED)
        for entry in self.generated_text_contains_everything_from_ground_truth_q_list:
            entry.configure(state=DISABLED)
        for entry in self.generated_text_contains_additional_realistic_elements_q_list:
            entry.configure(state=DISABLED)

        return self.stance_q_list


    def change_visibility_of_other_options(self, value):
        if int(value) == 1 or int(value) == 2:
            for option in self.stance_makes_no_sense_because_q_list:
                option.configure(state=NORMAL)
            for option in self.generated_text_contains_everything_from_ground_truth_q_list:
                option.configure(state=DISABLED)
            for option in self.generated_text_contains_additional_realistic_elements_q_list:
                option.configure(state=DISABLED)
        elif int(value) == 3 or int(value) == 4:
            for option in self.stance_makes_no_sense_because_q_list:
                option.configure(state=DISABLED)
            for option in self.generated_text_contains_everything_from_ground_truth_q_list:
                option.configure(state=NORMAL)
            for option in self.generated_text_contains_additional_realistic_elements_q_list:
                option.configure(state=NORMAL)

    def display_if_stance_does_not_make_sense_explain_why(self, x_pos, x_diff, y_pos, y_diff):
        label_if_stance_does_not_make_sense_why = Label(gui, text="Why does stance not (really) make sense?", width=70,
                                                        font=("Times New Roman", 12, "bold"), anchor='w')
        label_if_stance_does_not_make_sense_why.place(x=x_pos, y=y_pos + 9 * y_diff)
        self.stance_makes_no_sense_because_q_list = []
        buttons_stance_makes_no_sense_because = [Checkbutton(gui,
                                                             text="The stance is wrong (e.g. the stance is pro but the explanation is con or vice versa)",
                                                             variable=self.opt_selected_why_does_stance_not_make_sense__stance_is_wrong,
                                                             font=("courier new", 10)),
                                                 Checkbutton(gui,
                                                             text="The generated text does not make sense or is not understandable",
                                                             variable=self.opt_selected_why_does_stance_not_make_sense__generated_text_is_not_understandable,
                                                             font=("courier new", 10)),
                                                 Checkbutton(gui,
                                                             text="The generated text does not correspond the topic",
                                                             variable=self.opt_selected_why_does_stance_not_make_sense__generated_text_other_than_topic,
                                                             font=("courier new", 10))]
        count = 10
        for button in buttons_stance_makes_no_sense_because:
            button.place(x=x_pos, y=y_pos + count * y_diff)
            self.stance_makes_no_sense_because_q_list.append(button)
            count += 1
        self.comment_why_does_stance_not_make_sense__comment = Text(gui, height=1, width=40, bg="light green")
        self.comment_why_does_stance_not_make_sense__comment.insert(END, COMMENT_DEFAULT_STRING)
        self.comment_why_does_stance_not_make_sense__comment.place(x=x_pos, y=y_pos + count * y_diff)
        self.stance_makes_no_sense_because_q_list.append(self.comment_why_does_stance_not_make_sense__comment)


    def display_alternative_text_contains_everything_and_realistic_elements(self, x_pos, x_diff, y_pos, y_diff):

        # generated text contains everything from ground truth?
        hover_text_everything_included = """
        Select "1" if there is not a single element from the original text in the alternative text.
        Select "2" if the coverage is roughly 1 to 50 % elements.
        Select "3" if there are more than 50 and less than 100 % elements from the original text in the alternative one.
        Select "4" if every element from the original text is in the alternative one. Note that the alternative text might have more elements which is not a contradiction to this choice.
        """

        label_generated_text_contains_all_from_ground_truth = Label(gui,
                                                                    text="Is every element from the original is in the alternative text? (Note the hover text)",
                                                                    width=70, font=("Times New Roman", 12, "bold"), anchor='w')
        label_generated_text_contains_all_from_ground_truth.place(x=x_pos, y=y_pos + 15 * y_diff)
        Hovertip(label_generated_text_contains_all_from_ground_truth, hover_text_everything_included)
        self.generated_text_contains_everything_from_ground_truth_q_list = []
        tmp_x_pos = x_pos
        for i in range(1, 5):
            radio_does_stance_make_sense = Radiobutton(gui, text=i, variable=self.opt_selected_generated_text_contains_all_from_original, value=i, font=("courier new", 10))
            self.generated_text_contains_everything_from_ground_truth_q_list.append(radio_does_stance_make_sense)
            radio_does_stance_make_sense.place(x=tmp_x_pos, y=y_pos + 16 * y_diff)
            Hovertip(radio_does_stance_make_sense, hover_text_everything_included)
            tmp_x_pos += 40
        self.comment_generated_text_contains_all_from_original = Text(gui, height=1, width=40, bg="light green")
        self.comment_generated_text_contains_all_from_original.insert(END, COMMENT_DEFAULT_STRING)
        self.comment_generated_text_contains_all_from_original.place(x=tmp_x_pos + 10, y=y_pos + 16 * y_diff)
        Hovertip(self.comment_generated_text_contains_all_from_original, hover_text_everything_included)
        self.generated_text_contains_everything_from_ground_truth_q_list.append(self.comment_generated_text_contains_all_from_original)


        # generated text contains meaningful elements
        hover_text_additional_elements_are_realistic = """
        Select "1" if you would have expected exactly the opposite, e.g., an absolute opponent of vaccination who advocates vaccinations.
        Select "2" if it is rather unlikely that this group of persons would say make statement although it is not excluded. This would also include statements like "I like vaccinations but I do not like vaccinations".
        Select "3" if it is rather likely that this group of person would make that statement although it is not for sure.
        Select "4" if you had expected these statements, e.g., an absolute opponent of vaccination who is absolutely against vaccinations.
        """

        label_generated_text_contains_addition_realistic_elements = Label(gui,
                                                                          text="Are the additional elements in the alternative text realistic for a user with these properties? (Note the hover text)",
                                                                          width=90, font=("Times New Roman", 12, "bold"), anchor='w')
        label_generated_text_contains_addition_realistic_elements.place(x=x_pos, y=y_pos + 17 * y_diff)
        Hovertip(label_generated_text_contains_addition_realistic_elements, hover_text_additional_elements_are_realistic)
        self.generated_text_contains_additional_realistic_elements_q_list = []
        tmp_x_pos = x_pos
        for i in range(1, 5):
            radio_does_stance_make_sense = Radiobutton(gui, text=i, variable=self.opt_selected_generated_text_contains_realistic_additional_elements, value=i, font=("courier new", 10))
            self.generated_text_contains_additional_realistic_elements_q_list.append(radio_does_stance_make_sense)
            radio_does_stance_make_sense.place(x=tmp_x_pos, y=y_pos + 18 * y_diff)
            tmp_x_pos += 40
            Hovertip(radio_does_stance_make_sense, hover_text_additional_elements_are_realistic)
        self.comment_generated_text_contains_additional_realistic_elements = Text(gui, height=1, width=40, bg="light green")
        self.comment_generated_text_contains_additional_realistic_elements.insert(END, COMMENT_DEFAULT_STRING)
        self.comment_generated_text_contains_additional_realistic_elements.place(y=y_pos + 18 * y_diff, x=tmp_x_pos + 10)
        Hovertip(self.comment_generated_text_contains_additional_realistic_elements, hover_text_additional_elements_are_realistic)
        self.generated_text_contains_additional_realistic_elements_q_list.append(self.comment_generated_text_contains_additional_realistic_elements)

gui = Tk()
gui.geometry("850x1200")
gui.title("SOCIAL - Stakeholder Opinion prediCtIon Annotation tooL")

map_id_annotated_question = {}
try:
    with open('annotated_questions.json') as file_annotated:
        annotated_content = json.load(file_annotated)
    annotated_questions_json = (annotated_content)
    for arg_id in annotated_questions_json:
        map_id_annotated_question[arg_id] = annotated_content[arg_id]
except:
    pass

with open('questions.json') as file:
    questions_json = json.load(file)

questions = []
for arg_id in questions_json:
    if arg_id not in map_id_annotated_question:
        questions.append(
            (arg_id,
             questions_json[arg_id]['question'],
             questions_json[arg_id]['user'],
             questions_json[arg_id]['stakeholder_group'],
             questions_json[arg_id]['predicted_stance'],
             questions_json[arg_id]['ground_truth_stance'],
             questions_json[arg_id]['ground_truth_explanation'],
             questions_json[arg_id]['generated_explanation'],
             questions_json[arg_id]['approach_name'],
             questions_json[arg_id]['minDimensions'],
             questions_json[arg_id]['bucket'])
        )


answers = []
quiz = Annotation()
gui.mainloop()

annotated_content = {}

for arg_id in map_id_annotated_question:
    annotated_content[arg_id] = {}
    annotated_content[arg_id]["question"] = map_id_annotated_question[arg_id]["question"]
    annotated_content[arg_id]["user"] = map_id_annotated_question[arg_id]["user"]
    annotated_content[arg_id]["stakeholder_group"] = map_id_annotated_question[arg_id]["stakeholder_group"]
    annotated_content[arg_id]["predicted_stance"] = map_id_annotated_question[arg_id]["predicted_stance"]
    annotated_content[arg_id]["ground_truth_stance"] = map_id_annotated_question[arg_id]["ground_truth_stance"]
    annotated_content[arg_id]["ground_truth_explanation"] = map_id_annotated_question[arg_id]["ground_truth_explanation"]
    annotated_content[arg_id]["generated_explanation"] = map_id_annotated_question[arg_id]["generated_explanation"]
    annotated_content[arg_id]["approach_name"] = map_id_annotated_question[arg_id]["approach_name"]
    annotated_content[arg_id]["minDimensions"] = map_id_annotated_question[arg_id]["minDimensions"]
    annotated_content[arg_id]["bucket"] = map_id_annotated_question[arg_id]["bucket"]
    annotated_content[arg_id]["annotator_score"] = map_id_annotated_question[arg_id]["annotator_score"]


for question, answer in zip(questions, answers):
    if answer[0][0] != 0:

        annotated_content[question[0]] = {}
        annotated_content[question[0]]["question"] = question[1]
        annotated_content[question[0]]["user"] = question[2]
        annotated_content[question[0]]["stakeholder_group"] = question[3]
        annotated_content[question[0]]["predicted_stance"] = question[4]
        annotated_content[question[0]]["ground_truth_stance"] = question[5]
        annotated_content[question[0]]["ground_truth_explanation"] = question[6]
        annotated_content[question[0]]["generated_explanation"] = question[7]
        annotated_content[question[0]]["approach_name"] = question[8]
        annotated_content[question[0]]["minDimensions"] = question[9]
        annotated_content[question[0]]["bucket"] = question[10]

        annotated_content[question[0]]["annotator_score"] = {}
        annotated_content[question[0]]["annotator_score"]["does_stance_make_sense"] = {}
        annotated_content[question[0]]["annotator_score"]["does_stance_make_sense"]['rating'] = {}
        annotated_content[question[0]]["annotator_score"]["does_stance_make_sense"]['rating']['value'] = answer[0][0]
        annotated_content[question[0]]["annotator_score"]["does_stance_make_sense"]['rating']['comment'] = answer[0][1]

        annotated_content[question[0]]["annotator_score"]["does_stance_make_sense"]['no'] = {}
        annotated_content[question[0]]["annotator_score"]["does_stance_make_sense"]['no']["why_does_stance_not_make_sense__stance_is_wrong"] = answer[1][0]
        annotated_content[question[0]]["annotator_score"]["does_stance_make_sense"]['no']["why_does_stance_not_make_sense__generated_text_is_not_understandable"] = answer[1][1]
        annotated_content[question[0]]["annotator_score"]["does_stance_make_sense"]['no']["why_does_stance_not_make_sense__generated_text_other_than_topic"] = answer[1][2]
        annotated_content[question[0]]["annotator_score"]["does_stance_make_sense"]['no']["why_does_stance_not_make_sense__comment"] = answer[1][3]

        annotated_content[question[0]]["annotator_score"]["does_stance_make_sense"]['yes'] = {}
        annotated_content[question[0]]["annotator_score"]["does_stance_make_sense"]['yes']["generated_text_contains_all_from_original"] = {}
        annotated_content[question[0]]["annotator_score"]["does_stance_make_sense"]['yes']["generated_text_contains_all_from_original"]['rating'] = answer[2][0]
        annotated_content[question[0]]["annotator_score"]["does_stance_make_sense"]['yes']["generated_text_contains_all_from_original"]['comment'] = answer[2][1]

        annotated_content[question[0]]["annotator_score"]["does_stance_make_sense"]['yes']["generated_text_contains_realistic_additional_elements"] = {}
        annotated_content[question[0]]["annotator_score"]["does_stance_make_sense"]['yes']["generated_text_contains_realistic_additional_elements"]['rating'] = answer[3][0]
        annotated_content[question[0]]["annotator_score"]["does_stance_make_sense"]['yes']["generated_text_contains_realistic_additional_elements"]['comment'] = answer[3][1]

with open('annotated_questions.json', 'w', encoding='utf-8', newline='\n') as file:
    json.dump(annotated_content, file, ensure_ascii=False, indent=4)
    file.flush()
    file.close()

