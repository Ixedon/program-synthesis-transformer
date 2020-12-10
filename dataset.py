import json
import os
import re
from json.decoder import JSONDecodeError

from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

from DataLoader import load_programs_json, tokenize_program, decode_program, NotCompiledError
from interpreter.code_lisp import load_lisp_units, str_to_type, compile_func


class DataSet:

    def __init__(self, batch_size, total_train_count, total_val_count):

        self.input_tokenizer: Tokenizer = Tokenizer(filters='')
        self.target_tokenizer: Tokenizer = Tokenizer(filters='')

        self.input_vocab_size = None
        self.target_vocab_size = None
        self.max_input_len = None

        self.total_train_count = total_train_count
        self.total_val_count = total_val_count

        self.max_target_length = None
        self.batch_size = batch_size

        self.tf_train_dataset, self.__train_tests, self.total_train_count = self.__create_dataset(
            "metaset3.train.jsonl", total_train_count, True
        )
        self.tf_val_dataset, self.__val_tests, self.total_val_count = self.__create_dataset(
            "metaset3.dev.jsonl", total_val_count
        )

        self.lips_units = load_lisp_units()

    def __fit_tokenizers(self, texts, programs):
        self.input_tokenizer.fit_on_texts(texts)
        self.input_vocab_size = len(self.input_tokenizer.word_index) + 1

        text_tensor = self.input_tokenizer.texts_to_sequences(texts)
        text_tensor = pad_sequences(text_tensor, padding="post")

        self.target_tokenizer.fit_on_texts(programs)

        programs_tensor = self.target_tokenizer.texts_to_sequences(programs)
        programs_tensor = pad_sequences(programs_tensor, padding='post')

        self.max_input_len = text_tensor.shape[1]
        self.max_target_length = programs_tensor.shape[1]
        self.target_vocab_size = len(self.target_tokenizer.word_index) + 1
        return text_tensor, programs_tensor

    def __tokenize_programs(self, texts, programs):
        text_tensor = self.input_tokenizer.texts_to_sequences(texts)
        text_tensor = pad_sequences(text_tensor, self.max_input_len, padding='post')

        programs_tensor = self.target_tokenizer.texts_to_sequences(programs)
        programs_tensor = pad_sequences(programs_tensor, self.max_target_length, padding='post')
        return text_tensor, programs_tensor

    def __create_dataset(self, dataset_name: str, max_total_count: int, fit_tokenizer=False) -> (Dataset, list, int):
        programs_data = load_programs_json(os.path.join("cleared_data", dataset_name))
        texts = [self.preprocess_sentence(w) for w in programs_data['text']]
        programs_tokenized = []
        ids = []
        return_types = []
        args = []
        for i in range(len(programs_data["short_tree"])):
            program, program_args = tokenize_program(programs_data["short_tree"][i], programs_data["args"][i])
            programs_tokenized.append('<start> ' + program + ' <end>')
            return_types.append(programs_data["return_type"][i])
            args.append(program_args)
            ids.append(i)
        if fit_tokenizer:
            texts, programs = self.__fit_tokenizers(texts, programs_tokenized)
        else:
            texts, programs = self.__tokenize_programs(texts, programs_tokenized)
        tensor_len = len(programs[:max_total_count])
        dataset = Dataset.from_tensor_slices((
            texts[:max_total_count], programs[:max_total_count],
            return_types[:max_total_count], args[:max_total_count],
            ids[:max_total_count]
        )).shuffle(tensor_len)
        return dataset.batch(self.batch_size, drop_remainder=True), programs_data['tests'], tensor_len

    # def __create_val_dataset(self) -> Dataset:
    #     programs = load_programs_json(os.path.join("cleared_data", "metaset3.dev.jsonl"), self.total_val_count)
    #     texts = [self.preprocess_sentence(w) for w in programs['text']]
    #     self.__val_tests = programs['tests']
    #     programs_tokenized = []
    #     ids = []
    #     return_types = []
    #     args = []
    #     for i in range(len(programs["short_tree"])):
    #         program, program_args = tokenize_program(programs["short_tree"][i], programs["args"][i])
    #         programs_tokenized.append('<start> ' + program + ' <end>')
    #         return_types.append(programs["return_type"][i])
    #         args.append(program_args)
    #         ids.append(i)
    #     texts, programs = self.__tokenize_programs(texts, programs_tokenized)
    #     tensor_len = len(programs[:self.total_val_count])
    #     dataset = Dataset.from_tensor_slices((
    #         texts[:self.total_val_count], programs[:self.total_val_count], return_types[:self.total_val_count],
    #         args[:self.total_val_count], ids[:self.total_val_count]
    #     )).shuffle(tensor_len)
    #     self.total_val_count = tensor_len
    #     return dataset.batch(self.batch_size, drop_remainder=True)

    def get_target_index(self, word: str):
        return self.target_tokenizer.word_index[word]

    def get_target_word(self, index: int):
        return self.target_tokenizer.index_word[index]

    def get_train_count(self):
        return (self.total_train_count // self.batch_size) * self.batch_size

    def get_val_count(self):
        return (self.total_val_count // self.batch_size) * self.batch_size

    def take_train(self, steps_per_epoch):
        return self.tf_train_dataset.take(steps_per_epoch)

    def take_val(self, steps_per_epoch):
        return self.tf_val_dataset.take(steps_per_epoch)

    def take_train_tests(self, indices):
        return [self.__train_tests[i] for i in indices]

    def take_val_tests(self, indices):
        return [self.__val_tests[i] for i in indices]

    @staticmethod
    def preprocess_sentence(sentence):
        sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
        sentence = re.sub(r'[" ]+', " ", sentence)

        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)

        sentence = sentence.strip()
        sentence = '<start> ' + sentence + ' <end>'

        return sentence

    def preprocess_sequence(self, sentence):
        sentence = self.preprocess_sentence(sentence)
        tab = [self.input_tokenizer.word_index[i] for i in sentence.split()]
        while len(tab) < self.max_input_len:
            tab.append(0)
        return tab

    def decode_input(self, input_text) -> str:
        text = []
        for i in input_text.numpy():
            if i != 0:
                text.append(self.input_tokenizer.index_word[i])
        if "<start>" in text:
            text.remove("<start>")
        if "<end>" in text:
            text.remove("<end>")
        return " ".join(text)

    def get_program_tokens(self, encoded_program):
        program = []
        for i in encoded_program.numpy():
            if i != 0:
                program.append(self.target_tokenizer.index_word[i])
        if "<start>" in program:
            program.remove("<start>")
        if "<end>" in program:
            end_index = program.index("<end>")
            return program[:end_index]
        return program

    def decode_program(self, encoded_program, program_args):
        return decode_program(" ".join(self.get_program_tokens(encoded_program)), program_args)

    def compile_func(self, program, args, return_type):
        program = program.replace('"', '\\"')
        program = program.replace("'", '"')
        try:
            program = json.loads(program)
            args = [(key, str_to_type(args[key])) for key in args.keys()]
            return_type = str_to_type(return_type)
            return compile_func(self.lips_units, "program", program, args, return_type)
        except JSONDecodeError as e:
            raise NotCompiledError(e.args[0])
        except ValueError as e:
            raise NotCompiledError(e.args[0])
        except IndexError as e:
            raise NotCompiledError(e.args[0])
        except TypeError as e:
            raise NotCompiledError(e.args[0])
        except Exception as e:
            print(e.args)
            print(program)
            exit(-50)