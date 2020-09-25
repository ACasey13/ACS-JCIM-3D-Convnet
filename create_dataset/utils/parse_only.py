#!/usr/bin/env python3

import os
import argparse
import sys

try:
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
except KeyError:
    user_paths = []

SOURCE_PATH = '../source'
if os.path.abspath(SOURCE_PATH) not in [os.path.abspath(path) for path in user_paths]:
    sys.path.append(os.path.abspath(SOURCE_PATH))

from get_edat_prediction import gaussian_output


parser = argparse.ArgumentParser(description="parse output of given 'input_density.out' file")
parser.add_argument("filepath", help="file path to 'input_density.out' file to be parsed")
args = parser.parse_args()

print('parsing {}...'.format(args.filepath))
print('-'*50)
gaussian_output(args.filepath)
