import argparse

from DataSetClearer import DataSetClearer


def setup_parser() -> argparse.ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Algolips data set path", required=True)
    parser.add_argument("--output", help="Output path", default="cleared_data")
    return parser


if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args()
    DataSetClearer(args.dataset, args.output).clear_data()