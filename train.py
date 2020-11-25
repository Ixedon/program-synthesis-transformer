import tensorflow as tf

from model import Seq2Seq
from dataset import DataSet

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


dataset = DataSet(8)
model = Seq2Seq(128, 256, dataset)

model.train(10)
model.load_last()
print(model.evaluate_sentence("given an array of numbers what is first half of the given array"))