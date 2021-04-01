import torch, torch.nn as nn
from dataloader import WikiDataset
from model import SkipGram
import time
import numpy as np
import random
import itertools

def train(model, dataloader, optimizer, sampler, vocab_size, epochs=50, k = 2):
    model.train()
    for epoch in range(epochs):
        print(f'{"-"*10} Epoch {epoch} {"-"*10}')
        avg_loss = 0
        start = time.time()
        # batches
        for i, v in enumerate(train_dataloader):
            optimizer.zero_grad()
            l = v.reshape(v.shape[0] * v.shape[1], v.shape[2])
            
            # target
            target = l[:,0].to(model.device)

            # context
            context = l[:,1].to(model.device)

            # generate negative samples
            negative_samples = torch.tensor(sampler.sample(l.shape[0], k)).to(model.device)
            
            # increment by 1 the values which are the same as the context (very ugly i know)
            negative_samples = (negative_samples + (negative_samples.T - context).bool().logical_not().T * 1) % VOCAB_SIZE
            
            # run model and get loss
            loss = model(target, context, negative_samples)
            avg_loss += loss.item()
            
            # update model
            loss.backward()
            optimizer.step()
        
        end = time.time()
        print(f'Finished epoch {epoch}, took {end - start}s, average loss is {avg_loss / i}')
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"{SAVE_FOLDER}cbow_{epoch}")
        # test(dev_dataloader)


class NegativeSampler:
    def __init__(self, filename, words_to_idx):
        # Load word frequency and calculate power of 3/4 of every number
        frequencies = {} # word -> frequency powered to 3/4
        total_sum = 0
        with open(f'{filename}') as f:
            for line in f:
                word, count = line.strip().split()
                word = words_to_idx[word]
                freq = int(count) ** (3/4)

                frequencies[word] = freq
                total_sum += freq

        table_size = 100000000 # 100M
        self.words = []
        for word, count in frequencies.items():
            count = int(count / total_sum * table_size)
            self.words.extend([word] * count) 

        self.words = np.array(self.words)
        
    def sample(self, batch, k):
        words = np.random.randint(len(self.words), size=batch * k)
        return self.words[words].reshape((batch, k))


if __name__ == '__main__':
    EMBEDDING_DIM = 200
    BATCH_SIZE = 8192
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    FILENAME = "tmp.txt"
    SAVE_FOLDER = 'models/'
    
    with open(f'{FILENAME}.vocab') as f:
        vocab = sorted(f.readline().split())
    VOCAB_SIZE = len(vocab)

    # String2int conversion
    words_to_idx = {i:j for j,i in enumerate(vocab)}

    # Create dataset
    train_dataset = WikiDataset(f'{FILENAME}.parsed', 2, words_to_idx)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Get random sampler
    sampler = NegativeSampler(f'{FILENAME}.parsed.count', words_to_idx)
    
    model = SkipGram(VOCAB_SIZE, EMBEDDING_DIM)
    model.to(DEVICE)
    model.device = DEVICE

    optimizer = torch.optim.SparseAdam(model.parameters())

    train(model, train_dataloader, optimizer, sampler, VOCAB_SIZE)

