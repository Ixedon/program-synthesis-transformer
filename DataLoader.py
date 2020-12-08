import json
import os

from tqdm import tqdm


def load_programs_json(path, number=None):
    data_dict = {
        "text": [],
        "short_tree": [],
        "args": [],
        "return_type": [],
        "tests": []
    }
    with open(path, "r") as file:
        lines = file.readlines()
        file.close()
        if number:
            n = number
        else:
            n = len(lines)
        for line in tqdm(lines[:n]):
            ob = json.loads(line)
            data_dict["text"].append(" ".join(ob['text']))
            data_dict["short_tree"].append(ob["short_tree"])
            data_dict["args"].append(ob["args"])
            data_dict["return_type"].append(ob["return_type"])
            data_dict["tests"].append(ob['tests'])
    return data_dict


TYPES = {
    'any': '<any>',
    'int': '<int>',
    'string': '<string>',
    'int[]': '<int-array>',
    'string[]': '<string-array>',
    'bool': '<bool>',
    'int[][]': '<int-matrix>',
    'dict<int,int>': '<dict-int>',
    'dict<string,int>': '<dict-string>',
}

REV_TYPES = {v: k for k, v in TYPES.items()}


def type_to_token(str_type):
    return TYPES[str_type]


def token_to_type(token):
    return REV_TYPES[token]


def tokenize_program(program_tree, program_args):
    return encode_command(program_tree), encode_args(program_args)


# def encode_return_type(return_type):
#     return_types = ["<start-type>"]
#     return_types.append(type_to_token(return_type))
#     return_types.append("<end-type>")
#     return " ".join(return_types)


def encode_args(args):
    # args_tokens = ["<start-args>"]
    args_tokens = []
    for key in args.keys():
        args_tokens.append("<start-arg>")
        args_tokens.append(key)
        args_tokens.append(type_to_token(args[key]))
        args_tokens.append("<end-arg>")
    # args_tokens.append("<end-args>")
    return " ".join(args_tokens)


def encode_command(command: list):
    words = ["<l-bracket>"]
    for i in command:
        if isinstance(i, str):
            words.append(i)
        elif isinstance(i, list):
            words.append(encode_command(i))
        else:
            print(f"Error: type{type(i)}, {i}")
            raise ValueError("Unexpected type")
    words.append("<r-bracket>")
    return " ".join(words)


def decode_program(text, args_tokens):
    tokens = text.split()
    # TODO think about strings
    try:
        # return_type = token_to_type(tokens[tokens.index("<start-type>") + 1: tokens.index("<end-type>")][0])
        # args_tokens = tokens[tokens.index("<start-args>") + 1: tokens.index("<end-args>")]
        print("Args: args")
        args = decode_args(args_tokens.split())
        # program_start = text.index('<end-args>') + 11
        command = decode_command(text)#[program_start:])
        return command, args#, return_type
    except ValueError as e:
        raise NotCompiledError(e.args[0])
    except KeyError as e:
        raise NotCompiledError(e.args[0])
    except IndexError as e:
        raise NotCompiledError(e.args[0])


def decode_args(tokens):
    args = dict()
    while tokens:
        start_index = tokens.index("<start-arg>") + 1
        end_index = tokens.index("<end-arg>")
        arg = tokens[start_index: end_index][0]
        arg_type = token_to_type(tokens[start_index: end_index][1])
        args[arg] = arg_type
        tokens = tokens[end_index + 1:]
    return args


def split_program_into_tokens(program):
    tokens = []
    while program:
        # print(f"'{program}'")
        end_index = program.find(" ")
        if end_index == -1:
            # print(f"end:'{program}'")
            tokens.append(program)
            break
        if program[0] == '"':
            # print(f"prod:'{program[1:]}'")
            end_index = program[1:].index('"') + 2
        # print(f"Token:'{program[:end_index]}'")
        tokens.append(program[:end_index])
        program = program[end_index + 1:]
    return tokens


def decode_command(program):
    tokens = split_program_into_tokens(program)
    command = ""
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "<l-bracket>":
            command += "["
        elif token == "<r-bracket>":
            command += "]"
        else:
            command += f"'{token}'"
        if i + 1 < len(tokens):
            if tokens[i + 1] != "<r-bracket>" and token != '<l-bracket>':
                command += ", "
        i += 1
    return command


class NotCompiledError(Exception):

    def __init__(self, message):
        self.message = message
        super(NotCompiledError, self).__init__(f"Compilation Error: {message}")


if __name__ == '__main__':
    file_path = os.path.join("cleared_data", "metaset3.train.jsonl")
    programs = load_programs_json(file_path)
    for i in tqdm(range(len(programs["short_tree"]))):
        tree = programs["short_tree"][i]
        args = programs["args"][i]
        return_type = programs["return_type"][i]
        tokenized, tokenized_args = tokenize_program(tree, args)
        # print(tokenized)
        # exit(0)
        decoded_tree, decoded_args = decode_program(tokenized, tokenized_args)
        if str(tree) != decoded_tree:
            print(tokenized)
            print(tree)
            print(decoded_tree)
            raise ValueError("Decoding program error")
        if str(args) != str(decoded_args):
            print(tokenized)
            print(args)
            print(decoded_args)
            raise ValueError("Decoding args error")
        # if return_type != decoded_type:
        #     print(tokenized)
        #     print(return_type)
        #     print(decoded_type)
        #     raise ValueError("Decoding type")