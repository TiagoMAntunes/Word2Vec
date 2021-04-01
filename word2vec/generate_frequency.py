import sys
from collections import defaultdict
import csv

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <parsed file>")
        sys.exit(0)

    count = defaultdict(int) # word -> count

    with open(sys.argv[1]) as f:
        for word in f:
            # each line is a word
            count[word.strip()] += 1


    with open(f'{sys.argv[1]}.count', 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        for word, count in count.items():
            writer.writerow([word, count])

    
