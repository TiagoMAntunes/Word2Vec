"""
Processes the input data
"""

import sys
from pprint import pprint as pprint
import csv, os
from util import clean_line

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


emb_size = 2

f = open(sys.argv[1], 'r')

# i don't have enough memory to process this in one go
line = clean_line(f.readline())
vocab = set(line)

line_count = 0

def get_new_examples(line):
    res = [tuple(line[i-emb_size:i+emb_size+1]) for i in range(emb_size, len(line) - emb_size)]
    global line_count
    line_count += len(res)
    return res

def write_examples(exp, filename):
    with open(filename, "a+") as f:
        writer = csv.writer(f, delimiter=' ')
        for e in exp:
            writer.writerow(e)

EXAMPLES_FILE = f'{sys.argv[1]}.examples'
EXAMPLES_TRAIN = EXAMPLES_FILE + '.train'
EXAMPLES_DEV = EXAMPLES_FILE + '.dev'
EXAMPLES_TEST = EXAMPLES_FILE + '.test'



write_examples(get_new_examples(line), EXAMPLES_FILE)

prev = line[-emb_size:]

for line in f:
    # get independent words from the raw_text
    line = clean_line(line)
    vocab.update(set(line))
    # write to file
    write_examples(get_new_examples(prev + line), EXAMPLES_FILE)
    
f.close()

with open(f'{sys.argv[1]}.vocab', 'w+') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerow(vocab)

SEED = 669

# generate randomized splits
from sklearn.model_selection import train_test_split
print(f'Number of lines: {line_count}')
train, test = train_test_split(range(line_count), train_size=0.99, random_state=SEED, shuffle=True)
train, dev = train_test_split(train, train_size=0.99, random_state=SEED, shuffle=True)

# sort the data
train.sort()
dev.sort()
test.sort()

train_f = open(EXAMPLES_TRAIN, 'w+')
dev_f = open(EXAMPLES_DEV, 'w+')
test_f = open(EXAMPLES_TEST, 'w+')

train_i = 0
dev_i = 0
test_i = 0
# read the file line by line and output to the correct output file
print(len(train), len(dev), len(test))

with open(EXAMPLES_FILE) as f:
    for i, line in enumerate(f):
        if train_i < len(train) and i == train[train_i]:
            train_i += 1
            writer = train_f
        elif dev_i < len(dev) and i == dev[dev_i]:
            dev_i += 1
            writer = dev_f
        else:
            test_i += 1
            writer = test_f

        writer.write(line)

print(train_i, dev_i, test_i)
# close files
train_f.close()
dev_f.close()
test_f.close()

# delete big file
os.remove(EXAMPLES_FILE)
