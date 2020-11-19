import json
import os
import sys
import traceback

from interpreter.code_lisp import load_lisp_units, str_to_type, compile_func


class DataSetClearer:

    def __init__(self, dataset_path, output_path):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.lips_units = load_lisp_units()

        self.errors_quantity = 0
        self.errors_numbers = dict()
        self.not_passed = 0

    def clear_data(self):
        self.errors_numbers = []
        self.errors_quantity = 0
        programs_lines = self.__read_programs()
        f_name = os.path.basename(self.dataset_path)
        cleared_path = os.path.join(self.output_path, f_name)
        sys.setrecursionlimit(15000)
        with open(cleared_path, "w") as file:
            i = 0
            while programs_lines:
                print(i)
                line = programs_lines[0]
                program = json.loads(line)
                args = [(key, str_to_type(program['args'][key])) for key in program['args'].keys()]
                return_type = str_to_type(program['return_type'])
                statement = compile_func(self.lips_units, "test", program['short_tree'], args, return_type)
                tests = program['tests']
                ok_tests = []

                w_t = 0
                for j in range(len(tests)):
                    test_input = tests[j]['input']
                    test_output = tests[j]['output']

                    test_args = [test_input[a] for a in test_input.keys()]
                    try:
                        # print(j)
                        o = statement(*test_args)
                        if isinstance(o, range):
                            o = list(o)
                        if o != test_output:
                            w_t += 1
                        else:
                            ok_tests.append(tests[j])
                    except ValueError as e:
                        w_t += 1
                        print(f"Value error{i} {j}")
                        print("Text", " ".join(program['text']))
                        print("Args input:", program['args'])
                        print("Ret type:", program['return_type'])
                        print("Args:", *test_args)
                        print("Test output", test_output)
                        print("Tests", program['tests'])
                        print("Code:", program['short_tree'])
                        print(e)
                        if str(e) != "Computing HEAD of an empty array":
                            traceback.print_exc()
                        self.errors_numbers.append(i)
                    except RecursionError as e:
                        w_t += 1
                        print(f"RecursionError{i}  {j}")
                        print("Text", " ".join(program['text']))
                        print("Args input:", program['args'])
                        print("Ret type:", program['return_type'])
                        print("Args:", *test_args)
                        print("Test output", test_output)
                        print("Tests", program['tests'])
                        print("Code:", program['short_tree'])
                        traceback.print_exc()
                        self.errors_numbers.append(i)
                if w_t > 0:
                    print("error")
                    print("Text", " ".join(program['text']))
                    print("Args input:", program['args'])
                    print("Ret type:", program['return_type'])
                    print("Code:", program['short_tree'])
                    self.errors_quantity += 1
                if len(ok_tests) > 0:
                    program['tests'] = ok_tests
                    file.write(json.dumps(program) + "\n")
                else:
                    self.not_passed += 1
                del programs_lines[0]
                del program
                i += 1

        print(f"Path: {self.dataset_path}")
        print(f"Errors quantities: {self.errors_quantity}")
        print(f"Errors numbers: {self.errors_numbers}")
        print(f"Not passed: {self.not_passed}")

    def __read_programs(self):
        with open(self.dataset_path, "r") as file:
            lines = file.readlines()
            file.close()
        return lines
