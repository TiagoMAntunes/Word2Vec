"""
Processes the input data
"""

import sys
from pprint import pprint as print
import csv, os
from util import clean_line

if len(sys.argv) == 1:
    print(f"Usage: python3 {sys.argv[0]} <input data file>")
    sys.exit(0)

try:
    os.remove(f'{sys.argv[1]}.examples')
except OSError as error:
    print(error)

try:
    os.remove(f'{sys.argv[1]}.vocab')
except OSError as error:
    print(error)


emb_size = 2

f = open(sys.argv[1], 'r')

# i don't have enough memory to process this in one go
line = clean_line(f.readline())
vocab = set(line)

def get_new_examples(line):
    return [tuple(line[i-emb_size:i+emb_size+1]) for i in range(emb_size, len(line) - emb_size)]

def write_examples(exp):
    with open(f'{sys.argv[1]}.examples', "a+") as f:
        writer = csv.writer(f, delimiter=' ')
        for e in exp:
            writer.writerow(e)


write_examples(get_new_examples(line))

prev = line[-emb_size:]

for line in f:
    # get independent words from the raw_text
    line = clean_line(line)
    vocab.update(set(line))
    # write to file
    write_examples(get_new_examples(prev + line))
    
f.close()

with open(f'{sys.argv[1]}.vocab', 'w+') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerow(vocab)

# print(vocab)
# print(examples)
