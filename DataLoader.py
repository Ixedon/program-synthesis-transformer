import json
import os
import traceback

from tqdm import tqdm

from interpreter.code_lisp import compile_func, load_lisp_units
from interpreter.code_types import str_to_type

if __name__ == '__main__':
    file_path = os.path.join("cleared_data", "metaset3.dev.jsonl")
    with open(file_path, "r") as file:
        lines = file.readlines()
    units = load_lisp_units()
    errors = 0
    v_e = 0
    errors_numbers = []
    for j, line in enumerate(lines):
        ob = json.loads(line)
        # try:
        args = [(key, str_to_type(ob['args'][key])) for key in ob['args'].keys()]
        return_type = str_to_type(ob['return_type'])
        statement = compile_func(units, "test", ob['short_tree'], args, return_type)
        w_t = 0
        for i in range(len(ob['tests'])):

            test = ob['tests'][i]['input']
            o_test = ob['tests'][i]['output']

            test_args = [test[a] for a in test.keys()]
            o = statement(*test_args)
            if isinstance(o, range):
                o = list(o)
            if o != o_test:
                w_t += 1
        if w_t > 0:
            errors_numbers.append(j)
            print(f"error {j}")
            print("Text", " ".join(ob['text']))
            print("Args input:", ob['args'])
            print("Ret type:", ob['return_type'])
            print("Args:", *test_args)
            print("Output", o)
            print("Test output", o_test)
            print("Code:", ob['short_tree'])
            errors += 1
            print()
        if ob['tags']:
            print(ob['tags'])
    # except ValueError as e:
    #     print(f"Value error {j}")
    #     print("Text", " ".join(ob['text']))
    #     print("Args input:", ob['args'])
    #     print("Ret type:", ob['return_type'])
    #     print("Args:", *test_args)
    #     print("Output", o)
    #     print("Test output", o_test)
    #     print("Tests", ob['tests'])
    #     print("Code:", ob['short_tree'])
    #     print()
    #     errors_numbers.append(i)
    #     # traceback.print_exc()
    #     v_e += 1
    #     # exit(0)

    print("Errors:", errors)
    print("Ve:", v_e)
    # # statement.arg_values = [4,5]
    # print("Run",statement([4, 5, 4, 4]))

    print(errors_numbers)
