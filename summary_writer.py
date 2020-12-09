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
        self.train_writer = summary.create_file_writer(os.path.join(logs_dir, "train"))
        self.val_writer = summary.create_file_writer(os.path.join(logs_dir, "val"))

    def write_train_loss(self, train_loss, val_loss, step):
        with self.train_writer.as_default():
            summary.scalar("loss", train_loss, step)
        self.train_writer.flush()
        with self.val_writer.as_default():
            summary.scalar("loss", val_loss, step)
        self.val_writer.flush()

    def write_compiled_val_programs(self, compiled_percent, step):
        with self.val_writer.as_default():
            summary.scalar("compiled_programs", compiled_percent, step)
        self.val_writer.flush()

    def write_passed_test_count(self, passed_tests, step):
        with self.val_writer.as_default():
            summary.scalar("passed_tests", passed_tests, step)
        self.val_writer.flush()

    def write_generated_program(self, program, args, return_type, description, step):
        text = f"Args: {args}\n" \
               f"Program: {program}\n" \
               f"Return type: {return_type}"
        with self.val_writer.as_default():
            summary.text("program", text, step)
        self.val_writer.flush()
        with self.val_writer.as_default():
            summary.text("description", description, step)
        self.val_writer.flush()

    def write_mean_levenshtein_distance(self, train_levenshtein, val_levenshtein, step):
        with self.train_writer.as_default():
            summary.scalar("levenshtein", train_levenshtein, step)
        self.train_writer.flush()
        with self.val_writer.as_default():
            summary.scalar("levenshtein", val_levenshtein, step)
        self.val_writer.flush()
