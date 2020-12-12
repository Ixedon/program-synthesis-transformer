import datetime
import signal
import subprocess

import tensorflow as tf

from dataset import DataSet
from model import Seq2Seq
from summary_writer import TrainSummaryWriter

if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    dataset = DataSet(10, 80_000, 10_000, True)
    model = Seq2Seq(128, 320, dataset)
    summary_writer = TrainSummaryWriter("logs")
    model.set_summary_writer(summary_writer)

    logs_dir = "logs"
    current_date = datetime.date.today()
    current_date = current_date.strftime("%d-%m-%Y")
    logs_dir += "-" + current_date
    tensorboard_process = subprocess.Popen(f"tensorboard --logdir={logs_dir}")
    try:
        model.train(10, True)
        model.load_last("12-12-2020")
        print(model.evaluate_sentence("given an array of numbers what is first half of the given array"))
        tensorboard_process.send_signal(signal.CTRL_C_EVENT)
        print("Tensorboard killed")
    except Exception as e:
        tensorboard_process.send_signal(signal.CTRL_C_EVENT)
        tensorboard_process.kill()
        print("Tensorboard killed")
        raise e
