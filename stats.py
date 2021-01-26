import json
import os
import re

import tensorflow as tf
from tqdm import tqdm

from dataset import DataSet


def levenshtein_distance(truth, hyp):
    zeros = tf.zeros((truth.shape[0], 1), dtype=tf.int64)
    ranges = tf.reshape(tf.range(truth.shape[0], dtype=tf.int64), (truth.shape[0], 1))
    indices = tf.concat([zeros, ranges], axis=1)
    hyp_st = tf.SparseTensor(indices, hyp, [1, 1])
    truth_st = tf.SparseTensor(indices, truth, [1, 1])
    return tf.edit_distance(hyp_st, truth_st, normalize=False)


def encode_command(command: list):
    words = []
    for i in command:
        if isinstance(i, str):
            words.append(i)
        elif isinstance(i, list):
            words += encode_command(i)
            # words.append(encode_command(i))
        else:
            print(f"Error: type{type(i)}, {i}")
            raise ValueError("Unexpected type")
    return words

def load_programs_json(path, number=None):
    data_dict = {
        "text": [],
        "short_tree": []
    }
    with open(path, "r") as file:
        lines = file.readlines()
        file.close()
        if number:
            n = number
        else:
            n = len(lines)
        for line in tqdm(lines[:n], desc=f"Loading programs from jsonl file ({path})"):
            ob = json.loads(line)
            data_dict["text"].append(ob['text'])
            data_dict["short_tree"].append(ob["short_tree"])
    return data_dict


if __name__ == '__main__':
    # dataset = DataSet(10, 10, 10, True)
    train = load_programs_json(os.path.join("filtered_data", "metaset3.test.jsonl"))

    min_code = None
    code_sum = 0
    max_code = None

    min_text = None
    text_sum = 0
    max_text = None

    min_tokens = None
    tokens_sum = 0
    max_tokens = None
    count = 0
    for code, text in zip(train["short_tree"], train["text"]):
        encoded = encode_command(code)
        min_code = len(encoded) if min_code is None else min(min_code, len(encoded))
        code_sum += len(encoded)
        max_code = len(encoded) if max_code is None else max(max_code, len(encoded))

        min_tokens = len(text) if min_tokens is None else min(min_tokens, len(text))
        tokens_sum += len(text)
        max_tokens = len(text) if max_tokens is None else max(max_tokens, len(text))

        joined_text = " ".join(text)
        min_text = len(joined_text) if min_text is None else min(min_text, len(joined_text))
        text_sum += len(joined_text)
        max_text = len(joined_text) if max_text is None else max(max_text, len(joined_text))
        # print(text)
        count += 1
        # print(encoded)
        # exit(0)

    print(f"Min code: {min_code}")
    print(f"Mean code: {code_sum / count}")
    print(f"Max code: {max_code}")


    print(f"Min tokens: {min_tokens}")
    print(f"Mean tokens: {tokens_sum / count}")
    print(f"Max tokens: {max_tokens}")


    print(f"Min text: {min_text}")
    print(f"Mean text: {text_sum / count}")
    print(f"Max text: {max_text}")
