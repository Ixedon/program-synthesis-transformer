from dataset import DataSet
from model import Seq2Seq
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

dataset = DataSet(10, 80_000, 10_000, True)
model = Seq2Seq(128, 320, dataset)

model.load_last("17-12-2020")
# model.load_min_val_loss("17-12-2020")

model.test()

print(model.evaluate_sentence("given an array of numbers what is first half of the given array"))

