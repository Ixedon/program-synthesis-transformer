import json
import os
from DataLoader import load_programs_json
from argparse import ArgumentParser


def setup_parser() -> ArgumentParser():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--dataset", help="Algolips cleared data set path", required=True)
    arg_parser.add_argument("--output", help="Output path", default="filtered_data")
    return arg_parser


if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args()
    files = ["metaset3.test.jsonl", "metaset3.train.jsonl", "metaset3.dev.jsonl"]
    for file in files:

        programs = load_programs_json(os.path.join(args.dataset, file))
        filtered_programs = []
        j = 0
        for i in range(len(programs["short_tree"])):
            if "1000000000" not in json.dumps(programs["short_tree"][i]):
                j += 1
                filtered_programs.append(i)

        print(f"File {file}: {j}")
        with open(os.path.join(args.output, file), "w", encoding="utf-8") as f:
            lines = []
            for i in filtered_programs:
                lines.append(json.dumps({
                    "text": programs["text"][i].split(),
                    "short_tree": programs["short_tree"][i],
                    "args": programs["args"][i],
                    "return_type": programs["return_type"][i],
                    "tests": programs["tests"][i]
                }) + "\n")
            f.writelines(lines)
            f.close()
