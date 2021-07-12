import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, metavar='PATH', help='path to csv file')
    args = parser.parse_args()

    csv = pd.read_csv(args.csv)
        


