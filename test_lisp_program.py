import json

from interpreter.code_lisp import load_lisp_units, compile_func, str_to_type

if __name__ == '__main__':
    units = load_lisp_units()
    program_original = json.loads(
        "['digits', ['reduce', ['range', '0', ['+', 'a', '1']], '0', ['lambda2', ['+', ['*', 'arg1', '10'], 'arg2']]]]".replace(
            "'", '"'))
    # program_decoded = json.loads(
    #     "['slice', 'a', ['invoke1', ['lambda1', ['if', ['<=', 'arg1', '1'], '1', ['*', ['self', ['-', 'arg1', '1']], 'arg1']]], ['reduce', ['reverse', ['digits', 'b']], '0', ['lambda2', ['+', ['*', 'arg1', '10'], 'arg2']]]], 'c']".replace(
    #         "'", '"'))
    program = program_original
    program_args = json.loads('{"a": "int[]", "c": "int", "b": "int", "d": "int[]"}'.replace("'", '"'))
    args = [(key, str_to_type(program_args[key])) for key in program_args.keys()]
    return_type = str_to_type("int[]")
    test_input = json.loads(
        '{"a": [20, 29, 30, 16, 11, 28, 14, 28], "c": 5, "b": 5, "d": [15, 7, 7, 19, 3, 20, 30, 23, 25, 11, 12, 7, 5, 27]}'.replace(
            "'", '"'))
    statement = compile_func(units, "test", program, args, return_type)
    test_args = [test_input[a] for a in test_input.keys()]

    o = statement(*test_args)
    print(f"Output: {o}, type: {type(o)}")
