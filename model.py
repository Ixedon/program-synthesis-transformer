import datetime
import os
import sys
import time
import gc

import tensorflow as tf
import concurrent.futures as futures
from tensorflow import GradientTape
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from DataLoader import NotCompiledError, RunningTimeout
from dataset import DataSet
from summary_writer import TrainSummaryWriter


class Seq2Seq:

    def __init__(self, embeddig_dim, units, dataset: DataSet):
        super(Seq2Seq, self).__init__()
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

    def get_trainable_variables(self):
        return self.__encoder.trainable_variables + self.__decoder.trainable_variables

    def encode(self, x, hidden):
        return self.__encoder(x, hidden)

    def decode(self, x, hidden, encoder_output):
        return self.__decoder(x, hidden, encoder_output)

    def evaluate_sentence(self, sentence: str):
        input = tf.convert_to_tensor([self.__dataset.preprocess_sequence(sentence)])
        hidden = [tf.zeros((1, self.__units))]
        encoded_words = self.call(input, hidden)
        words = [self.__dataset.get_target_word(i) for i in encoded_words]
        return " ".join(words)

    def call(self, x, hidden):
        encoder_output, encoder_hidden = self.encode(x, hidden)

        decoder_hidden = encoder_hidden
        decoder_input = tf.expand_dims([self.__dataset.get_target_index('<start>')] * x.shape[0], 1)

        encoded_words = []
        for i in range(self.__dataset.max_target_length):
            predictions, decoder_hidden, attention_weights = self.decode(decoder_input, decoder_hidden, encoder_output)
            attention_weights = tf.reshape(attention_weights, (-1,))
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

    # TODO add not compiled loss
    def loss_function(self, target, prediction, compilation_mask):
        # if not compilation_mask:
        # print("Target", target.shape)
        # print("Prediction", prediction.shape)
        mask = tf.math.logical_not(tf.math.equal(target, 0))
        loss = self.__loss(target, prediction)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        loss *= compilation_mask
        return tf.reduce_mean(loss)

    # TODO add levenshtein loss
    def calculate_loss(self, predictions, target, compilation_mask):
        batch_loss = 0
        for i in range(1, target.shape[1]):
            # print("Pred shape before:", predictions[:, i, :].shape)
            pred = tf.reshape(predictions[:, i, :], (self.__batch_size, self.__dataset.target_vocab_size))
            # print("Pred shape after:", pred.shape)
            batch_loss += self.loss_function(target[:, i], pred, compilation_mask)
        return batch_loss

    @tf.function
    def train_step(self, input, target, encoder_hidden):
        loss = 0
        predicted = tf.zeros((self.__batch_size, 1), dtype=tf.int64)
        predictions_collection = tf.zeros((self.__batch_size, 1, self.__dataset.target_vocab_size), dtype=tf.float32)
        encoder_output, encoder_hidden = self.encode(input, encoder_hidden)

        decoder_hidden = encoder_hidden
        decoder_input = tf.expand_dims([self.__dataset.get_target_index('<start>')] * self.__batch_size, 1)
        for i in range(1, target.shape[1]):
            predictions, decoder_hidden, _ = self.decode(decoder_input, decoder_hidden, encoder_output)

            # TODO calculate loss for all targets and add not compilation error
            # loss += self.loss_function(target[:, i], predictions, tf.ones(self.__batch_size))

            decoder_input = tf.expand_dims(target[:, i], 1)

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

        # print("Pred collection:", predictions_collection.shape)

        # TODO for levensgtein
        # predicted = tf.cast(tf.reshape(predicted, targ.shape), tf.int32)
        # levenstein_loss = whole_loss(tf.cast(targ, tf.int32), predicted)[0]
        # tf.print("Loss", loss, output_stream=sys.stderr)
        # loss += levenstein_loss
        predicted = tf.cast(tf.reshape(predicted, target.shape), tf.int32)
        batch_loss = (loss / int(target.shape[1]))

        # gradients = tape.gradient(loss, self.get_trainable_variables())
        # self.__optimizer.apply_gradients(zip(gradients, self.get_trainable_variables()))

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
        executor = futures.ThreadPoolExecutor(max_workers=1)
        for epoch in range(epochs):
            start_time = time.time()
            encoder_hidden = self.get_initial_hidden_state()
            epoch_loss = 0

            compiled_programs = 0
            passed_tests = 0
            total_tests = 0
            first = True
            for (batch, (text, target, return_types, args, ids)) in enumerate(
                    self.__dataset.take_train(steps_per_epoch)
            ):
                if epoch == 0 and batch == 0:
                    print(f"Output tensor shape {target.shape}")
                with GradientTape() as tape:

                    predictions, predicted = self.train_step(text, target, encoder_hidden)
                    compilation_loss_mask = tf.ones(1)
                    if compile_train:
                        train_tests = self.__dataset.take_train_tests(ids)
                        for i, program in enumerate(predicted):
                            if first:
                                description = self.__dataset.decode_input(text[i])
                            else:
                                description = None
                            print(f"ProgramId: {ids[i]}")
                            future = executor.submit(self.evaluate_program, epoch, program, args[i], return_types[i],
                                                     train_tests[i], False, description)
                            try:
                                compiled, passed = future.result(60)
                                #             print(program)
                                executor._threads.clear()
                                futures.thread._threads_queues.clear()
                                #             return program, args
                            except futures.TimeoutError:
                                print("Timeout error", file=sys.stderr)
                                executor._threads.clear()
                                futures.thread._threads_queues.clear()
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
                    # print("Comp mask", compilation_loss_mask)
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

            # if (epoch + 1) % 2 == 0:
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
        first = True
        executor = futures.ThreadPoolExecutor(max_workers=1)
        for (batch, (input, target, return_types, args, ids)) in enumerate(self.__dataset.take_val(steps_per_epoch)):
            predictions, predicted = self.val_step(input, target, encoder_hidden)
            validation_tests = self.__dataset.take_val_tests(ids)
            compilation_loss_mask = tf.ones(1)
            for i, program in enumerate(predicted):
                if first:
                    description = self.__dataset.decode_input(input[i])
                else:
                    description = None

                print(f"ProgramId:{ids[i]}")
                future = executor.submit(self.evaluate_program, epoch, program, args[i], return_types[i],
                                         validation_tests[i], False, description)
                try:
                    compiled, passed = future.result(180)
                    #             print(program)
                    executor._threads.clear()
                    futures.thread._threads_queues.clear()
                    #             return program, args
                except futures.TimeoutError:
                    print("Timeout error", file=sys.stderr)
                    executor._threads.clear()
                    futures.thread._threads_queues.clear()
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
        self.__summary_writer.write_passed_test_count(passed_tests, epoch, True)
        self.__summary_writer.write_compiled_programs(compiled_programs, epoch, True)
        return val_loss

    def val_step(self, input, target, encoder_hidden):
        predicted = tf.zeros((self.__batch_size, 1), dtype=tf.int64)
        encoder_output, encoder_hidden = self.encode(input, encoder_hidden)
        predictions_collection = tf.zeros((self.__batch_size, 1, self.__dataset.target_vocab_size), dtype=tf.float32)

        decoder_hidden = encoder_hidden
        decoder_input = tf.expand_dims([self.__dataset.get_target_index('<start>')] * self.__batch_size, 1)

        for i in range(1, target.shape[1]):
            predictions, decoder_hidden, _ = self.decode(decoder_input, decoder_hidden, encoder_output)

            decoder_input = tf.expand_dims(target[:, i], 1)
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

        predicted = tf.cast(tf.reshape(predicted, target.shape), tf.int32)
        # TODO add levenshtein
        # levenstein_loss = whole_loss(tf.cast(targ, tf.int32), predicted)[0]
        # tf.print("Loss", loss, output_stream=sys.stderr)
        # loss += levenstein_loss
        # batch_loss = (loss / int(target.shape[1]))
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
            # print(self.__dataset.get_program_tokens(encoded_program))
            print("Prog", program)
            statement = self.__dataset.compile_func(program, args, return_type)
            no_error = True
            for i in range(len(tests)):
                test_input = tests[i]['input']
                test_output = tests[i]['output']
                test_args = [test_input[a] for a in test_input.keys()]
                # try:
                #     with futures.ThreadPoolExecutor(max_workers=1) as executor:
                #         future = executor.submit(statement, *test_args)
                #         try:
                o = statement(*test_args)
                # o = statement(*test_args)
                if isinstance(o, range):
                    o = list(o)
                if o == test_output:
                    passed_tests += 1
                    #     executor._threads.clear()
                    #     futures.thread._threads_queues.clear()
                    # except futures.TimeoutError:
                    #     executor._threads.clear()
                    #     futures.thread._threads_queues.clear()
                    #     raise RunningTimeout(program)
                # except RunningTimeout as e:
                #     no_error = False
                #     print(f"Test error: Running timeout {e.program}")
                #     break
                # except ValueError as e:
                #     no_error = False
                #     print(f"Tests error: {e.args[0]}", file=sys.stderr)
                #     break
                # except TypeError as e:
                #     no_error = False
                #     print(f"Tests error: {e.args[0]}", file=sys.stderr)
                #     break
                # except IndexError as e:
                #     no_error = False
                #     print(f"Tests error: {e.args[0]}", file=sys.stderr)
                #     break
                # except AttributeError as e:
                #     no_error = False
                #     print(f"Tests error: {e.args[0]}", file=sys.stderr)
                #     break
                # except KeyError as e:
                #     no_error = False
                #     print(f"Tests error: {e.args[0]}", file=sys.stderr)
                #     break
                # except MemoryError as e:
                #     no_error = False
                #     print(e)
                #     break
                # except AssertionError as e:
                #     no_error = False
                #     print(f"Tests error: {e}", file=sys.stderr)
                #     break
                # except Exception as e:
                #     no_error = False
                #     print(f"Other Tests error: {e}", file=sys.stderr)
                #     break
            del statement
            del program
            gc.collect()
            # print(f"PassedTests: {passed_tests} Total:{len(tests)}")
            compiled = 1 if no_error else 0
            print("Return")
            return compiled, passed_tests
        except Exception as e:
            # except NotCompiledError as e:
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
