#!/usr/bin/env python3

import argparse
import pandas as pd

default_h5 = '../non_volatile_data/cands-concat-clean.h5'

parser = argparse.ArgumentParser(description='Print .h5 file to screen')
parser.add_argument("filepath", nargs='?', 
                    help="filepath to the .h5 file to be printed to the string",
                    default=default_h5)
parser.add_argument("-n", "--n_lines", 
                    help="""number of lines to print.
specify a number. defaults to 'all'.""",
                    default='all')
args = parser.parse_args()  

try:
    df = pd.read_hdf(args.filepath)
except Exception as e:
    print(f'Error opening file {args.filepath}. Exception: {e}')
    raise

if args.n_lines == 'all':
    print(df)
else:
    print(df.head(int(args.n_lines)))
