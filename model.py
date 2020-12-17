import datetime
import os
import sys
import time
import gc

import tensorflow as tf
import concurrent.futures as futures
import numpy as np
from tensorflow import GradientTape
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from dataset import DataSet
from stats import levenshtein_distance
from summary_writer import TrainSummaryWriter
from timelimit import TimeoutException, time_limit


class Seq2Seq:

    def __init__(self, embeddig_dim, units, dataset: DataSet):
        super(Seq2Seq, self).__init__()
        tf.random.set_seed(0)
        np.random.seed(0)
        self.__batch_size = dataset.batch_size
        self.__units = units

        # Dataset
        self.__dataset = dataset

        print(f"Input Vocab size: {dataset.input_vocab_size}")
        # Encoder
        self.__encoder = Encoder(dataset.input_vocab_size, embeddig_dim, self.__units)

        print(f"Output Vocab size: {self.__dataset.target_vocab_size}")
        # Decoder
        self.__decoder = Decoder(self.__dataset.target_vocab_size, embeddig_dim, self.__units)

        self.__end_index = dataset.get_target_index('<end>')
        self.__optimizer = Adam()
        self.__loss = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        self.__summary_writer: TrainSummaryWriter = None
        self.__p_target = None

    def write_summary(self):
        input_layer = [Input(self.__dataset.max_input_len), Input(1)]
        encoder_model = self.__encoder.model(input_layer, self.get_initial_hidden_state())
        encoder_model.summary()
        enc_output, enc_hidden = encoder_model.output_shape
        print(f"Encoder hidden: {enc_hidden}")
        print(f"Encoder output: {enc_output}\n\n")
        self.__decoder.model(input_layer, hidden=encoder_model.output[1], enc_output=encoder_model.output[0]).summary()
        print("\n\n")

    def get_trainable_variables(self):
        return self.__encoder.trainable_variables + self.__decoder.trainable_variables

    def encode(self, x, hidden):
        return self.__encoder(x, hidden=hidden)

    def decode(self, x, hidden, encoder_output):
        return self.__decoder(x, hidden=hidden, enc_output=encoder_output)

    def evaluate_sentence(self, sentence: str):
        text_vector = tf.convert_to_tensor([self.__dataset.preprocess_sequence(sentence)])
        hidden = [tf.zeros((1, self.__units))]
        encoded_words = self.call(text_vector, hidden)
        words = [self.__dataset.get_target_word(i) for i in encoded_words]
        return " ".join(words)

    def call(self, x, hidden):
        encoder_output, encoder_hidden = self.encode(x, hidden)

        decoder_hidden = encoder_hidden
        decoder_input = tf.expand_dims([self.__dataset.get_target_index('<start>')] * x.shape[0], 1)

        encoded_words = []
        for i in range(self.__dataset.max_target_length):
            predictions, decoder_hidden, _ = self.decode(decoder_input, decoder_hidden, encoder_output)
            # attention_weights = tf.reshape(attention_weights, (-1,))
            # attention_plot[i] = attention_weights.numpy()
            predicted_id = tf.argmax(predictions[0]).numpy()
            encoded_words.append(predicted_id)
            decoder_input = tf.expand_dims([predicted_id], 0)
            if self.__end_index == predicted_id:
                return encoded_words  # , attention_plot

        return encoded_words  # , attention_plot

    def get_initial_hidden_state(self):
        return tf.zeros((self.__batch_size, self.__units))

    def load_last(self, date: str) -> None:
        checkpoint_dir = 'training_checkpoints-' + date
        checkpoint = tf.train.Checkpoint(
            optimizer=self.__optimizer,
            encoder=self.__encoder,
            decoder=self.__decoder
        )
        status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        status.assert_existing_objects_matched()

    def loss_function(self, target, prediction, compilation_mask):
        mask = tf.math.logical_not(tf.math.equal(target, 0))
        loss = self.__loss(target, prediction)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        loss *= compilation_mask
        return tf.reduce_mean(loss)

    def calculate_loss(self, predictions, target, compilation_mask):
        batch_loss = 0
        for i in range(1, target.shape[1]):
            pred = tf.reshape(predictions[:, i, :], (self.__batch_size, self.__dataset.target_vocab_size))
            batch_loss += self.loss_function(target[:, i], pred, compilation_mask)
        return batch_loss

    @tf.function
    def train_step(self, encoded_text, target, encoder_hidden, p):
        predicted = tf.zeros((self.__batch_size, 1), dtype=tf.int64)
        predictions_collection = tf.zeros((self.__batch_size, 1, self.__dataset.target_vocab_size), dtype=tf.float32)
        encoder_output, encoder_hidden = self.encode(encoded_text, encoder_hidden)

        decoder_hidden = encoder_hidden
        decoder_input = tf.expand_dims([self.__dataset.get_target_index('<start>')] * self.__batch_size, 1)
        for i in range(1, target.shape[1]):
            predictions, decoder_hidden, _ = self.decode(decoder_input, decoder_hidden, encoder_output)

            predicted_id = tf.argmax(predictions, axis=1)
            predicted_id = tf.reshape(predicted_id, (self.__batch_size, 1))
            predicted = tf.concat([predicted, predicted_id], axis=1)
            predictions_collection = tf.concat(
                [
                    predictions_collection,
                    tf.reshape(predictions, (self.__batch_size, 1, self.__dataset.target_vocab_size))
                ],
                axis=1
            )

            if p:  # tf.cond(tf.random.uniform((1,), minval=0, maxval=1) < p):
                decoder_input = tf.expand_dims(target[:, i], 1)
            else:
                decoder_input = tf.reshape(predicted_id, decoder_input.shape)

        predicted = tf.cast(tf.reshape(predicted, target.shape), tf.int32)

        return predictions_collection, predicted

    def train(self, epochs: int, compile_train: bool) -> None:
        current_date = datetime.date.today()
        current_date = current_date.strftime("%d-%m-%Y")
        print("Starting training")
        print(f"Train examples:{self.__dataset.get_train_count()}")
        print(f"Val examples:{self.__dataset.get_val_count()}")
        checkpoint_dir = 'training_checkpoints-' + current_date
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(
            optimizer=self.__optimizer,
            encoder=self.__encoder,
            decoder=self.__decoder
        )

        min_loss_checkpoint_dir = "min_loss_checkpoint-" + current_date
        min_loss_checkpoint_prefix = os.path.join(min_loss_checkpoint_dir, "ckpt")
        min_loss_checkpoint = tf.train.Checkpoint(
            optimizer=self.__optimizer,
            encoder=self.__encoder,
            decoder=self.__decoder
        )
        min_loss = 10_0000

        min_val_loss_checkpoint_dir = "min_val_loss_checkpoint-" + current_date
        min_val_loss_checkpoint_prefix = os.path.join(min_val_loss_checkpoint_dir, "ckpt")
        min_val_loss_checkpoint = tf.train.Checkpoint(
            optimizer=self.__optimizer,
            encoder=self.__encoder,
            decoder=self.__decoder
        )
        min_val_los = 10_000
        val_steps_per_epoch = self.__dataset.get_val_count() // self.__batch_size
        steps_per_epoch = self.__dataset.get_train_count() // self.__batch_size
        self.__p_target = tf.concat([tf.linspace(0.95, 0.3, epochs // 2), tf.zeros(epochs // 2)], axis=0)
        print(f"P tar {self.__p_target}")
        # executor = futures.ThreadPoolExecutor(max_workers=1)
        for epoch in range(epochs):
            start_time = time.time()
            encoder_hidden = self.get_initial_hidden_state()
            epoch_loss = 0

            compiled_programs = 0
            passed_tests = 0
            levenshtein_sum = 0
            equal_programs = 0
            total_tests = 0
            first = True
            for (batch, (text, target, return_types, args, ids)) in enumerate(
                    self.__dataset.take_train(steps_per_epoch)
            ):
                if epoch == 0 and batch == 0:
                    print(f"Output tensor shape {target.shape}")
                with GradientTape() as tape:
                    b = (tf.random.uniform([1], minval=0, maxval=1) < self.__p_target[epoch]).numpy()[0]
                    predictions, predicted = self.train_step(text, target, encoder_hidden, b)
                    compilation_loss_mask = tf.ones(1)
                    if compile_train:
                        train_tests = self.__dataset.take_train_tests(ids)
                        for i, program in enumerate(predicted):
                            dist = levenshtein_distance(target[i], program)
                            if dist == 0:
                                equal_programs += 1
                            levenshtein_sum += dist
                            if first:
                                description = self.__dataset.decode_input(text[i])
                            else:
                                description = None
                            print(f"ProgramId: {ids[i]}")
                            # future = executor.submit(self.evaluate_program, epoch, program, args[i], return_types[i],
                            #                          train_tests[i], False, description)
                            try:
                                with time_limit(180):
                                    compiled, passed = self.evaluate_program(epoch, program, args[i], return_types[i],
                                                                             train_tests[i], False, description)
                                # executor._threads.clear()
                                # futures.thread._threads_queues.clear()
                            except TimeoutException:
                                print("Timeout error", file=sys.stderr)
                                # executor._threads.clear()
                                # futures.thread._threads_queues.clear()
                                compiled = 0
                                passed = 0

                            if compiled == 0:
                                compilation_loss_mask = tf.concat([compilation_loss_mask, [2]], axis=0)
                            else:
                                compilation_loss_mask = tf.concat([compilation_loss_mask, [1]], axis=0)

                            passed_tests += passed
                            compiled_programs += compiled
                            total_tests += len(train_tests[i])
                            first = False
                        compilation_loss_mask = compilation_loss_mask[1:]
                    batch_loss = self.calculate_loss(predictions, target, compilation_loss_mask)
                    batch_loss = (batch_loss / int(target.shape[1]))

                    gradients = tape.gradient(batch_loss, self.get_trainable_variables())
                    self.__optimizer.apply_gradients(zip(gradients, self.get_trainable_variables()))

                    epoch_loss += batch_loss

                del predictions
                del compilation_loss_mask

                if batch % 100 == 0:
                    print(
                        'Epoch {} Batch {} Loss {:.4f}'.format(
                            epoch + 1,
                            batch,
                            batch_loss
                        )
                    )

            if compile_train:
                passed_tests = passed_tests / total_tests
                print(f"Train Compiled: {compiled_programs} Count: {self.__dataset.get_train_count()}")
                compiled_programs = compiled_programs / self.__dataset.get_train_count()
                self.__summary_writer.write_passed_test_count(passed_tests, epoch, False)
                self.__summary_writer.write_compiled_programs(compiled_programs, epoch, False)

            mean_levenshtein = levenshtein_sum / self.__dataset.get_train_count()
            mean_equality = equal_programs / self.__dataset.get_train_count()
            self.__summary_writer.write_mean_equality(mean_equality, epoch, False)
            self.__summary_writer.write_mean_levenshtein_distance(np.asscalar(mean_levenshtein.numpy()), epoch, False)

            checkpoint.save(file_prefix=checkpoint_prefix)

            val_loss = self.validate(epoch)
            if val_loss / val_steps_per_epoch < min_val_los:
                min_val_los = val_loss
                min_val_loss_checkpoint.save(file_prefix=min_val_loss_checkpoint_prefix)
            self.__summary_writer.write_train_loss(epoch_loss / steps_per_epoch, val_loss / val_steps_per_epoch, epoch)
            if epoch_loss / steps_per_epoch < min_loss:
                min_loss_checkpoint.save(file_prefix=min_loss_checkpoint_prefix)
                min_loss = epoch_loss / steps_per_epoch
            print('Epoch {} Loss {:.4f} Val loss: {:.4f}'.format(epoch + 1, epoch_loss / steps_per_epoch, val_loss))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start_time))

    def validate(self, epoch: int) -> float:
        val_loss = 0
        steps_per_epoch = self.__dataset.get_val_count() // self.__batch_size
        encoder_hidden = self.get_initial_hidden_state()
        compiled_programs = 0
        passed_tests = 0
        total_tests = 0
        levenshtein_sum = 0
        equal_programs = 0
        first = True
        # executor = futures.ThreadPoolExecutor(max_workers=1)
        for (batch, (text, target, return_types, args, ids)) in enumerate(self.__dataset.take_val(steps_per_epoch)):
            predictions, predicted = self.val_step(text, target, encoder_hidden)
            validation_tests = self.__dataset.take_val_tests(ids)
            compilation_loss_mask = tf.ones(1)
            for i, program in enumerate(predicted):
                dist = levenshtein_distance(target[i], program)
                if dist == 0:
                    equal_programs += 1
                levenshtein_sum += dist
                if first:
                    description = self.__dataset.decode_input(text[i])
                else:
                    description = None

                print(f"ProgramId:{ids[i]}")
                # future = executor.submit(self.evaluate_program, epoch, program, args[i], return_types[i],
                #                          validation_tests[i], True, description)
                try:
                    with time_limit(180):
                        compiled, passed = self.evaluate_program(epoch, program, args[i], return_types[i],
                                                                 validation_tests[i], True, description)
                    #             print(program)
                    # executor._threads.clear()
                    # futures.thread._threads_queues.clear()
                    #             return program, args
                except TimeoutException:
                    print("Timeout error", file=sys.stderr)
                    # executor._threads.clear()
                    # futures.thread._threads_queues.clear()
                    compiled = 0
                    passed = 0

                if compiled == 0:
                    compilation_loss_mask = tf.concat([compilation_loss_mask, [2]], axis=0)
                else:
                    compilation_loss_mask = tf.concat([compilation_loss_mask, [1]], axis=0)
                passed_tests += passed
                compiled_programs += compiled
                total_tests += len(validation_tests[i])
                first = False
            compilation_loss_mask = compilation_loss_mask[1:]
            batch_loss = self.calculate_loss(predictions, target, compilation_loss_mask)
            batch_loss = (batch_loss / int(target.shape[1]))
            val_loss += batch_loss
            del predictions
            del predicted
            gc.collect()
            if batch % 100 == 0:
                print(
                    'Validation: Batch {} Loss {:.4f}'.format(
                        batch,
                        batch_loss
                    )
                )

        passed_tests = passed_tests / total_tests
        print(f"Compiled: {compiled_programs} Count: {self.__dataset.get_val_count()}")
        compiled_programs = compiled_programs / self.__dataset.get_val_count()
        mean_levenshtein = levenshtein_sum / self.__dataset.get_val_count()
        mean_equality = equal_programs / self.__dataset.get_val_count()
        self.__summary_writer.write_passed_test_count(passed_tests, epoch, True)
        self.__summary_writer.write_compiled_programs(compiled_programs, epoch, True)
        self.__summary_writer.write_mean_equality(mean_equality, epoch, True)
        self.__summary_writer.write_mean_levenshtein_distance(np.asscalar(mean_levenshtein.numpy()), epoch, True)
        return val_loss

    @tf.function
    def val_step(self, input, target, encoder_hidden):
        predicted = tf.zeros((self.__batch_size, 1), dtype=tf.int64)
        encoder_output, encoder_hidden = self.encode(input, encoder_hidden)
        predictions_collection = tf.zeros((self.__batch_size, 1, self.__dataset.target_vocab_size), dtype=tf.float32)

        decoder_hidden = encoder_hidden
        decoder_input = tf.expand_dims([self.__dataset.get_target_index('<start>')] * self.__batch_size, 1)

        for i in range(1, target.shape[1]):
            predictions, decoder_hidden, _ = self.decode(decoder_input, decoder_hidden, encoder_output)
            predicted_id = tf.argmax(predictions, axis=1)
            decoder_input = tf.reshape(predicted_id, decoder_input.shape)

            predicted_id = tf.reshape(predicted_id, (self.__batch_size, 1))
            predicted = tf.concat([predicted, predicted_id], axis=1)

            predictions_collection = tf.concat(
                [
                    predictions_collection,
                    tf.reshape(predictions, (self.__batch_size, 1, self.__dataset.target_vocab_size))
                ],
                axis=1
            )

        predicted = tf.cast(tf.reshape(predicted, target.shape), tf.int32)
        return predictions_collection, predicted

    def evaluate_program(self, epoch: int, encoded_program, program_args, program_return_type, tests,
                         is_validation: bool,
                         description: str = None) -> (int, int, int):
        written = False
        passed_tests = 0
        try:
            program, args = self.__dataset.decode_program(encoded_program, program_args)
            if description:
                self.__summary_writer.write_generated_program(program, args, program_return_type, description, epoch,
                                                              is_validation)
                written = True
            return_type = program_return_type.numpy().decode("utf-8")
            print("Prog", program)
            statement = self.__dataset.compile_func(program, args, return_type)
            no_error = True
            for i in range(len(tests)):
                test_input = tests[i]['input']
                test_output = tests[i]['output']
                test_args = [test_input[a] for a in test_input.keys()]
                o = statement(*test_args)
                if isinstance(o, range):
                    o = list(o)
                if o == test_output:
                    passed_tests += 1
            del statement
            del program
            gc.collect()
            # print(f"PassedTests: {passed_tests} Total:{len(tests)}")
            compiled = 1 if no_error else 0
            print("Return")
            return compiled, passed_tests
        except Exception as e:
            print(f"Error:{e}", file=sys.stderr)
            if description and not written:
                text = " ".join(self.__dataset.get_program_tokens(encoded_program))
                self.__summary_writer.write_generated_program(text, f"CompilationError: {e}", "", description,
                                                              epoch, is_validation)
            return 0, passed_tests

    def set_summary_writer(self, summary_writer: TrainSummaryWriter):
        self.__summary_writer = summary_writer


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(Encoder, self).__init__(name="Encoder")
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, inputs, **kwargs):  # x, hidden):
        hidden = kwargs.get("hidden")
        x = self.embedding(inputs)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def model(self, input_layer, hidden):
        return tf.keras.Model(input_layer, self.call(input_layer[0], hidden=hidden), name="Encoder")


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

    def call(self, inputs, **kwargs):  # , hidden, enc_output):
        hidden = kwargs.get("hidden")
        enc_output = kwargs.get("enc_output")
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(inputs)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

    def model(self, input_layer, hidden, enc_output):
        return tf.keras.Model(input_layer, self.call(input_layer[1], hidden=hidden, enc_output=enc_output),
                              name="Decoder")


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__(name="BahdanauAttention")
        self.dense_0 = Dense(units)
        self.dense_1 = Dense(units)
        self.dense_2 = Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)

        score = self.dense_2(tf.nn.tanh(
            self.dense_0(query_with_time_axis) + self.dense_1(values)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
