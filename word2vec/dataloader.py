import torch
import numpy as np

class WikiDataset(torch.utils.data.IterableDataset):
    """
        Wrapper for the iterable WikiData
        file must be a word-per-line type
        e.g.:
            potato
            orange
            apple
    """
    def __init__(self, file, word_range, words_to_idx):        
        self.file = file
        self.word_range = word_range
        self.words_conversion = words_to_idx

    def __iter__(self):
        return WikiData(self.file, self.word_range, self.words_conversion)

class WikiData:
    def __init__(self, filename, word_range, conversion):
        self.file = open(filename)
        self.words = []
        self.word_range = word_range
        self.words_to_idx = conversion

    def __next__(self):
        """
            Returns multiple (target, context) pairs for the same target
        """
        while len(self.words) <= self.word_range * 2 + 1:
            line = self.file.readline().strip()
            if not line:
                raise StopIteration()
            self.words.append(self.words_to_idx[line])

        res = np.array([(self.words[self.word_range], self.words[self.word_range+i]) for i in (-2,-1,1,2)])
        # advance window
        self.words = self.words[1:]

        return res
        


if __name__ == '__main__':
    filename = 'tmp.txt'

    with open(f'{filename}.vocab') as f:
        vocab = sorted(f.readline().split())
    VOCAB_SIZE = len(vocab)

    # String2int conversion
    words_to_idx = {i:j for j,i in enumerate(vocab)}

    BATCH_SIZE = 8096
    NUM_WORKERS = 4
    # Create dataset
    train = WikiDataset(f'{filename}.parsed', 2, words_to_idx)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    for i, v in enumerate(train_dataloader):
        l = v.reshape(v.shape[0] * v.shape[1], v.shape[2])
        print(i, l.shape)
        
        

         