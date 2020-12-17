from argparse import ArgumentParser

from dataset import DataSet
from model import Seq2Seq


def setup_parser() -> ArgumentParser():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--summary", help="Prints out model summary", dest="summary", action="store_true")
    return arg_parser


def print_summary():
    dataset = DataSet(10, 10, 10, False)
    model = Seq2Seq(128, 320, dataset)
    model.write_summary()


if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args()
    if args.summary:
        print_summary()
