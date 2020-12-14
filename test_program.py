import json
import concurrent.futures as futures
import time

from interpreter.code_lisp import load_lisp_units, compile_func, str_to_type


def infinite(n):
    i = n
    while True:
        # print(i)
        i += 1
        return 50


def main():
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(infinite, 50)
        try:
            resp = future.result(30)
        except futures.TimeoutError:
            print('Some other resp')
        else:
            print(resp)
        executor._threads.clear()
        futures.thread._threads_queues.clear()


if __name__ == '__main__':
    main()

    # units = load_lisp_units()
    # program_original = json.loads("['slice', 'a', ['invoke1', ['lambda1', ['if', ['<=', 'arg1', '1'], '1', ['*', ['self', ['-', 'arg1', '1']], 'arg1']]], ['reduce', ['reverse', ['digits', 'b']], '0', ['lambda2', ['+', ['*', 'arg1', '10'], 'arg2']]]], 'c']".replace("'", '"'))
    # program_decoded  = json.loads("['+', 'a', ['invoke1', ['lambda1', ['if', ['==', ['len', ['digits', 'arg1']], '1'], '0', ['+', '1', ['self', ['reduce', ['digits', 'arg1'], '0', '+']]]]], ['invoke1', ['lambda1', ['if', ['<=', 'arg1', '1'], '1', ['*', ['self', ['-', 'arg1', '1']], 'arg1']]], 'b']]]".replace("'", '"'))
    # program = program_original
    # program_args = json.loads("{'a': 'int[]', 'c': 'int', 'b': 'int'}".replace("'", '"'))
    # args = [(key, str_to_type(program_args[key])) for key in program_args.keys()]
    # return_type = str_to_type("int[]")
    # test_input = json.loads("{'a': [6, 28, 29, 4, 23, 19, 19, 15, 25, 29, 14, 11, 30, 17, 16, 25, 14, 27, 30, 12, 3, 22, 29, 5, 13, 20, 18, 26, 9, 29, 10, 15, 30, 30, 18, 16, 8, 8, 17, 5], 'c': 10, 'b': 4}".replace("'", '"'))
    # statement = compile_func(units, "test", program, args, return_type)
    # test_args = [test_input[a] for a in test_input.keys()]
    #
    # o = statement(*test_args)
    # print(f"Output: {o}, type: {type(o)}")

# I = 12246
# Original ['+', 'a', ['invoke1', ['lambda1', ['if', ['==', ['len', ['digits', 'arg1']], '1'], '0', ['+', '1', ['self', ['reduce', ['digits', 'arg1'], '0', '+']]]]], ['invoke1', ['lambda1', ['if', ['<=', 'arg1', '1'], '1', ['*', ['self', ['-', 'arg1', '1']], 'arg1']]], 'b']]]
# Encoded  + a invoke1 lambda1 if == len digits arg1 1 0 + 1 self reduce digits arg1 0 + invoke1 lambda1 if <= arg1 1 1 * self - arg1 1 arg1 b
# Decoded  ['+', 'a', ['invoke1', ['lambda1', ['if', ['==', ['len', ['digits', 'arg1']], '1'], '0', ['+', '1', ['self', ['reduce', ['digits', 'arg1'], '0', '+']]]]], ['invoke1', ['lambda1', ['if', ['<=', 'arg1', '1'], '1', ['*', ['self', ['-', 'arg1', '1']], 'arg1']]], 'b']]]
# Program args {'a': 'int', 'b': 'int'}
# Return type int
# Input {'a': 19, 'b': 10}
# Output 20
# Expected 21
# Test fallen
# Original ['+', 'a', ['invoke1', ['lambda1', ['if', ['==', ['len', ['digits', 'arg1']], '1'], '0', ['+', '1', ['self', ['reduce', ['digits', 'arg1'], '0', '+']]]]], ['invoke1', ['lambda1', ['if', ['<=', 'arg1', '1'], '1', ['*', ['self', ['-', 'arg1', '1']], 'arg1']]], 'b']]]
# Encoded  + a invoke1 lambda1 if == len digits arg1 1 0 + 1 self reduce digits arg1 0 + invoke1 lambda1 if <= arg1 1 1 * self - arg1 1 arg1 b
# Decoded  ['+', 'a', ['invoke1', ['lambda1', ['if', ['==', ['len', ['digits', 'arg1']], '1'], '0', ['+', '1', ['self', ['reduce', ['digits', 'arg1'], '0', '+']]]]], ['invoke1', ['lambda1', ['if', ['<=', 'arg1', '1'], '1', ['*', ['self', ['-', 'arg1', '1']], 'arg1']]], 'b']]]
