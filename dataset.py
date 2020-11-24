import os
import re

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.data import Dataset

from DataLoader import load_programs_json, tokenize_program


class DataSet:

    def __init__(self, batch_size):
        self.input_tokenizer: Tokenizer = None
        self.target_tokenizer: Tokenizer = None
        self.tf_dataset: Dataset = None
        self.input_vocab_size = None
        self.target_vocab_size = None
        self.total_count = None
        self.max_target_length = None
        self.batch_size = batch_size
        self._create_dataset()

    def _tokenize(self, texts, programs):
        self.input_tokenizer = Tokenizer(filters='')
        self.input_tokenizer.fit_on_texts(texts)
        self.input_vocab_size = len(self.input_tokenizer.word_index) + 1

        text_tensor = self.input_tokenizer.texts_to_sequences(texts)
        text_tensor = pad_sequences(text_tensor, padding="post")

        self.target_tokenizer = Tokenizer(filters='')
        self.target_tokenizer.fit_on_texts(programs)

        programs_tensor = self.target_tokenizer.texts_to_sequences(programs)
        programs_tensor = pad_sequences(programs_tensor, padding='post')

        self.total_count = len(text_tensor)
        self.max_input_len = text_tensor.shape[1]
        self.max_target_length = programs_tensor.shape[1]
        self.target_vocab_size = len(self.target_tokenizer.word_index) + 1
        return text_tensor, programs_tensor

    def _create_dataset(self):
        programs = load_programs_json(os.path.join("cleared_data", "metaset3.train.jsonl"))
        texts = [self.preprocess_sentence(w) for w in programs['text']]
        programs_tokenized = []
        for i in range(len(programs["short_tree"])):
            programs_tokenized.append('<start> ' + tokenize_program(programs["short_tree"][i], programs["args"][i],
                                                                    programs["return_type"][i]) + ' <end>')
        texts, programs = self._tokenize(texts, programs_tokenized)
        tensor_len = len(programs)
        dataset = Dataset.from_tensor_slices((texts, programs)).shuffle(tensor_len)
        self.tf_dataset = dataset.batch(self.batch_size, drop_remainder=True)

    def get_target_index(self, word: str):
        return self.target_tokenizer.word_index[word]

    def get_target_word(self, index: int):
        return self.target_tokenizer.index_word[index]

    def get_train_count(self):
        return self.total_count

    def take(self, steps_per_epoch):
        return self.tf_dataset.take(steps_per_epoch)

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
