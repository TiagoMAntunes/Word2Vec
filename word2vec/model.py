import torch
import torch.nn as nn
import sys
import numpy as np
import time
from util import clean_line

class CBOW(torch.nn.Module):
    
    
    def __init__(self, vocabulary_size, embedding_dimension):
        super(CBOW, self).__init__()
        
        self.embeddings = nn.Embedding(vocabulary_size, embedding_dimension)
        self.linear = nn.Linear(embedding_dimension, vocabulary_size)
        self.activate = nn.LogSoftmax(dim=-2)

    
    def forward(self, inputs):
        # get average embeddings
        embeddings = torch.mean(self.embeddings(inputs), 1)
        return self.activate(self.linear(embeddings))


class WikiDataset(torch.utils.data.IterableDataset):
    def __init__(self, file, word_range):        
        self.file = file
        self.word_range = word_range

    def __iter__(self):
        return WikiData(self.file, self.word_range)

class WikiData:
    def __init__(self, filename, word_range):
        self.file = open(filename)
        self.words = []
        self.word_range = word_range

    def __next__(self):
        """
            Returns a word_range *2 + 1 sized np.array for training
        """
        while len(self.words) < self.word_range * 2 + 1:
            line = self.file.readline()
            if not line:
                raise StopIteration()
                
            self.words.extend(list(clean_line(line)))
        words = tuple(map(lambda x: words_to_idx[x], self.words[:self.word_range * 2 + 1]))
        self.words = self.words[1:]

        mid = len(words) // 2
        return np.array([*words[:mid], *words[mid+1:]]), np.array(words[mid])

EMBEDDING_DIM = 200
BATCH_SIZE = 8192
NUM_WORKERS = 2
device = "cuda" if torch.cuda.is_available() else "cpu"

filename = 'tmp.txt'
SAVE_FOLDER = "models/"

with open(f'{filename}.vocab') as f:
    vocab = sorted(f.readline().split())
VOCAB_SIZE = len(vocab)

# String2int conversion
words_to_idx = {i:j for j,i in enumerate(vocab)}

# Create dataset
train = WikiDataset(f'{filename}', 2)
train_dataloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# Create model
model = CBOW(VOCAB_SIZE, EMBEDDING_DIM)
model.to(device)

loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

def train(train_dataloader):
    for epoch in range(50): 
        model.train()
        
        print(f'{"-"*10} Epoch {epoch} {"-"*10}')
        start = time.time()
        avg_loss = 0
        correct = 0
        count = 0
        # batches
        for i, (context_t, target_t)  in enumerate(train_dataloader):
            optimizer.zero_grad()

            context_t = context_t.to(device)
            target_t = target_t.to(device)
        
            # run model
            probs = model(context_t)

            # get loss
            loss = loss_fn(probs, target_t)

            results = torch.max(probs, dim=1)[1] == target_t
            correct += results.sum()
            count += len(results)

            avg_loss += loss.item()
            
            # update model
            loss.backward()
            optimizer.step()
        
        end = time.time()
        print(f'Finished epoch {epoch}, took {end - start}s, average loss is {avg_loss / i}, accuracy is {correct / count}')
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"{SAVE_FOLDER}cbow_{epoch}")
        # test(dev_dataloader)

train(train_dataloader)