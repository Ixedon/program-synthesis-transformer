import json
import os
import sys

from DataLoader import load_programs_json, NotCompiledError
from interpreter.code_lisp import load_lisp_units, str_to_type, compile_func
from tqdm import tqdm

additional_tokens = {
    '" "': "<space-string>"
}
rev_add_tokens = {v: k for k, v in additional_tokens.items()}

lambda_args = ["arg1", "arg2"]
self_args = ["self"]
partials = ["partial0", "partial1"]

DEBUG = False


def check_has_callable_args(command, lisp_units) -> bool:
    for arg in lisp_units[command].args:
        if callable(arg):
            return True
    return False


def encode_program_no_brackets(command):
    words = []
    for i in command:
        if isinstance(i, str):
            if i not in additional_tokens.keys():
                words.append(i)
            else:
                words.append(additional_tokens[i])
        elif isinstance(i, list):
            words.append(encode_program_no_brackets(i))
        else:
            print(f"Error: type{type(i)}, {i}")
            raise ValueError("Unexpected type")
    return " ".join(words)


def process_partial(partial_command, command_tokens, program_args, lisp_units):
    program = [partial_command]
    next_arg, command_tokens = decode_command_no_brackets(command_tokens, program_args, lisp_units)
    if next_arg:
        program.append(next_arg)
    current_command = command_tokens[0]
    if check_has_callable_args(current_command, lisp_units):
        next_arg, command_tokens = decode_command_no_brackets(command_tokens, program_args, lisp_units)
    else:
        next_arg = command_tokens[0]
        command_tokens = command_tokens[1:]
    if next_arg:
        program.append(next_arg)

    return program, command_tokens


def process_combine(command_tokens, program_args, lisp_units):
    program = ["combine"]
    current_command = command_tokens[0]
    if check_has_callable_args(current_command, lisp_units):
        next_arg, command_tokens = decode_command_no_brackets(command_tokens, program_args, lisp_units)
    else:
        next_arg = command_tokens[0]
        command_tokens = command_tokens[1:]
    if next_arg:
        program.append(next_arg)

    current_command = command_tokens[0]
    if check_has_callable_args(current_command, lisp_units):
        next_arg, command_tokens = decode_command_no_brackets(command_tokens, program_args, lisp_units)
    else:
        next_arg = command_tokens[0]
        command_tokens = command_tokens[1:]
    if next_arg:
        program.append(next_arg)

    return program, command_tokens


def process_filter(command_tokens, program_args, lisp_units):
    program = ["filter"]
    next_arg, command_tokens = decode_command_no_brackets(command_tokens, program_args, lisp_units)
    if next_arg:
        program.append(next_arg)

    current_command = command_tokens[0]
    if check_has_callable_args(current_command, lisp_units):
        next_arg, command_tokens = decode_command_no_brackets(command_tokens, program_args, lisp_units)
    else:
        next_arg = command_tokens[0]
        command_tokens = command_tokens[1:]
    if next_arg:
        program.append(next_arg)

    return program, command_tokens


def process_map(command_tokens, program_args, lisp_units):
    program = ["map"]
    next_arg, command_tokens = decode_command_no_brackets(command_tokens, program_args, lisp_units)
    if next_arg:
        program.append(next_arg)

    current_command = command_tokens[0]
    if check_has_callable_args(current_command, lisp_units):
        next_arg, command_tokens = decode_command_no_brackets(command_tokens, program_args, lisp_units)
    else:
        next_arg = command_tokens[0]
        command_tokens = command_tokens[1:]
    if next_arg:
        program.append(next_arg)

    return program, command_tokens


def process_reduce(command_tokens, program_args, lisp_units):
    program = ["reduce"]
    next_arg, command_tokens = decode_command_no_brackets(command_tokens, program_args, lisp_units)
    if next_arg:
        program.append(next_arg)

    next_arg, command_tokens = decode_command_no_brackets(command_tokens, program_args, lisp_units)
    if next_arg:
        program.append(next_arg)

    current_command = command_tokens[0]
    if check_has_callable_args(current_command, lisp_units):
        next_arg, command_tokens = decode_command_no_brackets(command_tokens, program_args, lisp_units)
    else:
        next_arg = command_tokens[0]
        command_tokens = command_tokens[1:]
    if next_arg:
        program.append(next_arg)

    return program, command_tokens


def decode_command_no_brackets(command_tokens, program_args, lisp_units, i):
    # time.sleep(10)
    if i >= 500:
        raise NotCompiledError("MaxRecursion")
    if not command_tokens:
        return None, command_tokens
    current_command = command_tokens[0]
    command_tokens = command_tokens[1:]
    if DEBUG:
        print("\n\n")
        print("Current command", current_command)
        print("CorrectCommandTokens", command_tokens)
    # If command is in special tokens
    if current_command in rev_add_tokens.keys():
        current_command = rev_add_tokens[current_command]
    # If command is program argument
    if current_command in program_args:
        if DEBUG:
            print("Returning program args", current_command)
        return current_command, command_tokens
    # If command is lambda argument
    if current_command in lambda_args:
        if DEBUG:
            print("Returning lambda_args", current_command)
        return current_command, command_tokens
    # If command is lambda

    # If command is partials
    if current_command in partials:
        program, command_tokens = process_partial(current_command, command_tokens, program_args, lisp_units)
        if DEBUG:
            print("Returning partial", program)
        return program, command_tokens

    # If command is combine
    if current_command == "combine":
        program, current_tokens = process_combine(command_tokens, program_args, lisp_units)
        if DEBUG:
            print("Returning combine", program)
        return program, current_tokens
    # If command is invoke

    # If command is filter:
    if current_command == "filter":
        program, command_tokens = process_filter(command_tokens, program_args, lisp_units)
        if DEBUG:
            print("Returning filter", program)
        return program, command_tokens
    # If command is reduce
    if current_command == "reduce":
        program, command_tokens = process_reduce(command_tokens, program_args, lisp_units)
        if DEBUG:
            print("Returning reduce", program)
        return program, command_tokens
    # If command is map
    if current_command == "map":
        program, command_tokens = process_map(command_tokens, program_args, lisp_units)
        if DEBUG:
            print("Returning map", program)
        return program, command_tokens

    program = []
    # If command is self
    if current_command in self_args:
        if current_command:
            program.append(current_command)
        next_arg, command_tokens = decode_command_no_brackets(command_tokens, program_args, lisp_units)
        if next_arg:
            program.append(next_arg)

        if DEBUG:
            print("Returning self_args", program)
        return program, command_tokens

    args_count = len(lisp_units[current_command].args)
    if DEBUG:
        print("Args count:", args_count)

    # If command has no args
    if args_count == 0:
        return current_command, command_tokens
    if current_command:
        program.append(current_command)

    i = 0
    while i < args_count:
        next_arg, command_tokens = decode_command_no_brackets(command_tokens, program_args, lisp_units)
        if DEBUG:
            print("While", command_tokens, "Current command", current_command)
        if next_arg:
            program.append(next_arg)
        i += 1
    if DEBUG:
        print("Returning:", program)
    return program, command_tokens


def main():
    file_path = os.path.join("cleared_data", "metaset3.train.jsonl")
    programs = load_programs_json(file_path)
    start = 0
    run = True
    sys.setrecursionlimit(15000)
    for i in tqdm(range(start, min(len(programs["short_tree"]), 120_000))):
        print(f"I = {i}")
        original_program = programs["short_tree"][i]

        args_tokens = programs["args"][i].keys()
        encoded = encode_program_no_brackets(original_program)
        decoded = decode_command_no_brackets(encoded.split(), args_tokens, units1)[0]

        print("Original", original_program)
        print("Encoded ", encoded)
        print("Decoded ", decoded)
        equal = json.dumps(programs["short_tree"][i]) == json.dumps(decoded)
        if not equal:
            print("Programs not equal")
            exit(-2)

        if run:
            program_args = programs['args'][i]
            args = [(key, str_to_type(program_args[key])) for key in program_args.keys()]
            return_type = str_to_type(programs["return_type"][i])
            statement = compile_func(units1, "test", decoded, args, return_type)

            tests = programs["tests"][i]
            for j in range(len(tests)):
                test_input = tests[j]["input"]
                test_output = tests[j]["output"]

                test_args = [test_input[a] for a in test_input.keys()]

                try:
                    output = statement(*test_args)
                    if isinstance(output, range):
                        output = list(output)
                    if output != test_output:
                        print("Program args", program_args)
                        print("Return type", programs["return_type"][i])
                        print("Input", test_input)
                        print("Output", output)
                        print("Expected", test_output)
                        print("Test fallen")
                        print("Original", original_program)
                        print("Encoded ", encoded)
                        print("Decoded ", decoded)
                        exit(-111)
                        # raise Exception("Wrong output")
                    del output
                except Exception as e:
                    print("Test fallen")
                    print("Original", original_program)
                    print("Encoded ", encoded)
                    print("Decoded ", decoded)
                    print("\n\n")
                    raise e
            del statement
        print("\n\n")


if __name__ == '__main__':
    units1 = load_lisp_units()
    main()
