import datetime
import os
import shutil

from tensorflow import summary


class TrainSummaryWriter:

    def __init__(self, logs_dir):
        current_date = datetime.date.today()
        current_date = current_date.strftime("%d-%m-%Y")
        logs_dir += "-" + current_date
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        if len(os.listdir(logs_dir)) > 0:
            shutil.rmtree(logs_dir)
        self.__train_writer = summary.create_file_writer(os.path.join(logs_dir, "train"))
        self.__val_writer = summary.create_file_writer(os.path.join(logs_dir, "val"))

    def write_train_loss(self, train_loss, val_loss, step: int):
        with self.__train_writer.as_default():
            summary.scalar("loss", train_loss, step)
        self.__train_writer.flush()
        with self.__val_writer.as_default():
            summary.scalar("loss", val_loss, step)
        self.__val_writer.flush()

    def write_compiled_programs(self, compiled_percent, step: int, is_validation: bool):
        if is_validation:
            with self.__val_writer.as_default():
                summary.scalar("compiled_programs", compiled_percent, step)
            self.__val_writer.flush()
        else:
            with self.__train_writer.as_default():
                summary.scalar("compiled_programs", compiled_percent, step)
            self.__train_writer.flush()

    def write_passed_test_count(self, passed_tests, step: int, is_validation: bool):
        if is_validation:
            with self.__val_writer.as_default():
                summary.scalar("passed_tests", passed_tests, step)
            self.__val_writer.flush()
        else:
            with self.__train_writer.as_default():
                summary.scalar("passed_tests", passed_tests, step)
            self.__train_writer.flush()

    def write_generated_program(self, program, args, return_type, description, step: int, is_validation: bool):
        text = f"Description: {description}<br>" \
               f"Args: {args}<br>" \
               f"Program: {program}<br>" \
               f"Return type: {return_type}"
        if is_validation:
            with self.__val_writer.as_default():
                summary.text("program", text, step)
            self.__val_writer.flush()
        else:
            with self.__train_writer.as_default():
                summary.text("program", text, step)
            self.__train_writer.flush()

    def write_mean_levenshtein_distance(self, levenshtein, step: int, is_validation: bool):
        if is_validation:
            with self.__val_writer.as_default():
                summary.scalar("levenshtein", levenshtein, step)
            self.__val_writer.flush()
        else:
            with self.__train_writer.as_default():
                summary.scalar("levenshtein", levenshtein, step)
            self.__train_writer.flush()

    def write_mean_equality(self, equality, step: int, is_validation: bool):
        if is_validation:
            with self.__val_writer.as_default():
                summary.scalar("equality", equality, step)
            self.__val_writer.flush()
        else:
            with self.__train_writer.as_default():
                summary.scalar("equality", equality, step)
            self.__train_writer.flush()

        # TODO add program equality, hemingway distance
