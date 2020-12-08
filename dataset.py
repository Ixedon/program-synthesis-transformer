import json
import os
import re

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.data import Dataset

from interpreter.code_lisp import load_lisp_units, str_to_type, compile_func
from DataLoader import load_programs_json, tokenize_program, decode_program


class DataSet:

    def __init__(self, batch_size, total_train_count, total_val_count):

        self.input_tokenizer: Tokenizer = None
        self.target_tokenizer: Tokenizer = None

        self.tf_train_dataset: Dataset = None
        self.tf_val_dataset: Dataset = None
        self.val_tests = None

        self.input_vocab_size = None
        self.target_vocab_size = None
        self.max_input_len = None

        self.total_train_count = total_train_count
        self.total_val_count = total_val_count

        self.max_target_length = None
        self.batch_size = batch_size
        self.__create_train_dataset()
        self.__create_val_dataset()
        self.lips_units = load_lisp_units()

    def __create_tokenizers(self, texts, programs):
        self.input_tokenizer = Tokenizer(filters='')
        self.input_tokenizer.fit_on_texts(texts)
        self.input_vocab_size = len(self.input_tokenizer.word_index) + 1

        text_tensor = self.input_tokenizer.texts_to_sequences(texts)
        text_tensor = pad_sequences(text_tensor, padding="post")

        self.target_tokenizer = Tokenizer(filters='')
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

    def __create_train_dataset(self):
        programs = load_programs_json(os.path.join("cleared_data", "metaset3.train.jsonl"))
        texts = [self.preprocess_sentence(w) for w in programs['text']]
        programs_tokenized = []
        ids = []
        return_types = []
        args = []
        for i in range(len(programs["short_tree"])):
            program, program_args = tokenize_program(programs["short_tree"][i], programs["args"][i])
            programs_tokenized.append('<start> ' + program + ' <end>')
            return_types.append(programs["return_type"][i])
            args.append(program_args)
            ids.append(i)
        texts, programs = self.__create_tokenizers(texts, programs_tokenized)
        tensor_len = len(programs[:self.total_train_count])
        dataset = Dataset.from_tensor_slices((
            texts[:self.total_train_count], programs[:self.total_train_count],
            return_types[:self.total_train_count], args[:self.total_train_count],
            ids[:self.total_train_count]
        )).shuffle(tensor_len)
        self.tf_train_dataset = dataset.batch(self.batch_size, drop_remainder=True)

    def __create_val_dataset(self):
        programs = load_programs_json(os.path.join("cleared_data", "metaset3.dev.jsonl"), self.total_val_count)
        texts = [self.preprocess_sentence(w) for w in programs['text']]
        programs_tokenized = []
        self.val_tests = programs['tests']
        ids = []
        return_types = []
        args = []
        for i in range(len(programs["short_tree"])):
            program, program_args = tokenize_program(programs["short_tree"][i], programs["args"][i])
            programs_tokenized.append('<start> ' + program + ' <end>')
            return_types.append(programs["return_type"][i])
            args.append(program_args)
            ids.append(i)
        texts, programs = self.__tokenize_programs(texts, programs_tokenized)
        tensor_len = len(programs[:self.total_val_count])
        dataset = Dataset.from_tensor_slices((
            texts[:self.total_val_count], programs[:self.total_val_count], return_types[:self.total_val_count],
            args[:self.total_val_count], ids[:self.total_val_count]
        )).shuffle(tensor_len)
        self.total_val_count = tensor_len
        self.tf_val_dataset = dataset.batch(self.batch_size, drop_remainder=True)

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

    def take_val_tests(self, indices):
        return [self.val_tests[i] for i in indices]

    def preprocess_sentence(self, sentence):
        sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)

        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)

        sentence = sentence.strip()
        sentence = '<start> ' + sentence + ' <end>'

        return sentence

    def preprocess_sequecnse(self, sentence):
        sentence = self.preprocess_sentence(sentence)
        tab = [self.input_tokenizer.word_index[i] for i in sentence.split()]
        while len(tab) < self.max_input_len:
            tab.append(0)
        return tab

    def decode_input(self, input) -> str:
        text = []
        for i in input.numpy():
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
        end_index = program.index("<end>")
        if "<start>" in program:
            program.remove("<start>")
        return program[:end_index]

    def decode_program(self, encoded_program, program_args):
        print(encoded_program.shape)
        tokens = self.get_program_tokens(encoded_program)
        print(tokens)
        return decode_program(" ".join(self.get_program_tokens(encoded_program)), program_args)

    def compile_func(self, program, args, return_type):
        program = program.replace('"', '\\"')
        program = program.replace("'", '"')
        program = json.loads(program)
        args = [(key, str_to_type(args[key])) for key in args.keys()]
        return_type = str_to_type(return_type)
        return compile_func(self.lips_units, "program", program, args, return_type)
