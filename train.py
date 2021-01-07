import datetime
import sys

import tensorflow as tf

from dataset import DataSet
from model import Seq2Seq
from summary_writer import TrainSummaryWriter


if __name__ == '__main__':
    tf.random.set_seed(0)
    sys.setrecursionlimit(15000)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    dataset = DataSet(10, 80_000, 10_000, True)
    model = Seq2Seq(128, 320, dataset)
    model.load_last("30-12-2020")
    summary_writer = TrainSummaryWriter("logs")
    model.set_summary_writer(summary_writer)

    logs_dir = "logs"
    current_date = datetime.date.today()
    current_date = current_date.strftime("%d-%m-%Y")
    logs_dir += "-" + current_date
    model.train(6, True)
    model.load_last(current_date)
    print(model.evaluate_sentence("given an array of numbers what is first half of the given array"))
