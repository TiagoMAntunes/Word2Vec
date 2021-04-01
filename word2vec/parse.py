"""
Processes the input data
"""

import sys
from pprint import pprint as pprint
import csv, os
from util import clean_line
import nltk, enchant
if len(sys.argv) == 1:
    print(f"Usage: python3 {sys.argv[0]} <input data file>")
    sys.exit(0)

try:
    os.remove(f'{sys.argv[1]}.examples')
except OSError as error:
    print(f'{sys.argv[1]}.examples', error)

try:
    os.remove(f'{sys.argv[1]}.vocab')
except OSError as error:
    print(f'{sys.argv[1]}.vocab', error)

vocab = set()
VOCAB_FILE = f'{sys.argv[1]}.vocab'
PARSED_FILE = f'{sys.argv[1]}.parsed'

with open(sys.argv[1]) as input_file, open(PARSED_FILE, "w+") as parsed_file:
    writer = csv.writer(parsed_file, delimiter='\n')
    for i, line in enumerate(input_file):
        if i % 100000 == 0:
            print(f"Iteration {i}")
        line = clean_line(line)
        if line:
            vocab.update(set(line))
            writer.writerow(line)
    

# write vocabulary
with open(f'{sys.argv[1]}.vocab', 'w+') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerow(vocab)
