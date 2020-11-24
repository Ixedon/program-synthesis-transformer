import os
import time

import tensorflow as tf
from tensorflow import GradientTape
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

import numpy as np

from dataset import DataSet


class Seq2Seq:

    def __init__(self, embeddig_dim, units, dataset: DataSet):
        super(Seq2Seq, self).__init__()
        self.batch_size = dataset.batch_size
        self.units = units

        # Dataset
        self.dataset = dataset

        # Encoder
        self.encoder = Encoder(dataset.input_vocab_size, embeddig_dim, self.units)

        # Decoder
        self.decoder = Decoder(self.dataset.target_vocab_size, embeddig_dim, self.units)

        self.end_index = dataset.get_target_index('<end>')
        self.optimizer = Adam()
        self.loss = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def get_trainable_variables(self):
        return self.encoder.trainable_variables + self.decoder.trainable_variables

    def loss_function(self, target, prediction):
        mask = tf.math.logical_not(tf.math.equal(target, 0))
        loss = self.loss(target, prediction)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_mean(loss)

    def encode(self, x, hidden):
        return self.encoder(x, hidden)

    def decode(self, x, hidden, encoder_output):
        return self.decoder(x, hidden, encoder_output)

    def evaluate_sentence(self, sentence: str):
        input = tf.convert_to_tensor([self.dataset.preprocess_sequecnse(sentence)])
        hidden = [tf.zeros((1, self.units))]
        encoded_words = self.call(input, hidden)
        words = [self.dataset.get_target_word(i) for i in encoded_words]
        return " ".join(words)

    def call(self, x, hidden):
        encoder_output, encoder_hidden = self.encode(x, hidden)

        decoder_hidden = encoder_hidden
        decoder_input = tf.expand_dims([self.dataset.get_target_index('<start>')] * x.shape[0], 1)

        encoded_words = []
        for i in range(self.dataset.max_target_length):
            predictions, decoder_hidden, attention_weights = self.decode(decoder_input, decoder_hidden, encoder_output)

            attention_weights = tf.reshape(attention_weights, (-1,))
            # attention_plot[i] = attention_weights.numpy()
            predicted_id = tf.argmax(predictions[0]).numpy()
            encoded_words.append(predicted_id)
            if self.end_index == predicted_id:
                return encoded_words  # , attention_plot
        return encoded_words  # , attention_plot

    @tf.function
    def train_step(self, input, target, encoder_hidden):
        loss = 0
        # TODO for levenstein
        # predicted = tf.zeros((self.batch_size, 1), dtype=tf.int64)
        with GradientTape() as tape:
            encoder_output, encoder_hidden = self.encode(input, encoder_hidden)

            decoder_hidden = encoder_hidden
            decoder_input = tf.expand_dims([self.dataset.get_target_index('<start>')] * self.batch_size, 1)
            for i in range(1, target.shape[1]):
                predictions, decoder_hidden, _ = self.decode(decoder_input, decoder_hidden, encoder_output)

                loss += self.loss_function(target[:, i], predictions)

                decoder_input = tf.expand_dims(target[:, i], 1)
            # TODO for levenstein
            # predicted_id = tf.argmax(predictions, axis=1)
            # predicted_id = tf.reshape(predicted_id, (8, 1))
            # predicted = tf.concat([predicted, predicted_id], axis=0)

        # TODO for levenstein
        # predicted = tf.cast(tf.reshape(predicted, targ.shape), tf.int32)
        # levenstein_loss = whole_loss(tf.cast(targ, tf.int32), predicted)[0]
        # tf.print("Loss", loss, output_stream=sys.stderr)
        # loss += levenstein_loss
        batch_loss = (loss / int(target.shape[1]))

        gradients = tape.gradient(loss, self.get_trainable_variables())
        self.optimizer.apply_gradients(zip(gradients, self.get_trainable_variables()))
        return batch_loss

    def get_initial_hidden_state(self):
        return tf.zeros((self.batch_size, self.units))

    def train(self, epochs):
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            encoder=self.encoder,
            decoder=self.decoder
        )

        steps_per_epoch = self.dataset.get_train_count() // self.batch_size
        for epoch in range(epochs):
            start_time = time.time()
            encoder_hidden = self.get_initial_hidden_state()
            epoch_loss = 0
            for (batch, (input, target)) in enumerate(self.dataset.take(steps_per_epoch)):
                batch_loss = self.train_step(input, target, encoder_hidden)
                epoch_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 batch_loss.numpy()))
            if (epoch + 1) % 2 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            print('Epoch {} Loss {:.4f}'.format(epoch + 1, epoch_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start_time))


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # Attention
        self.attention = BahdanauAttention(units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.dense_0 = Dense(units)
        self.dense_1 = Dense(units)
        self.dense_2 = Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.dense_2(tf.nn.tanh(
            self.dense_0(query_with_time_axis) + self.dense_1(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
