import torch
import torch.nn as nn
import sys
import numpy as np
import time

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


filename = 'tmp.txt'

with open(f'{filename}.vocab') as f:
    vocab = sorted(f.readline().split())

data = []
words_to_idx = {i:j for j,i in enumerate(vocab)}


# generate training set
with open(f'{filename}.examples') as f:
    for line in f:
        line = tuple(map(lambda x: words_to_idx[x], line.split()))
        mid = len(line) // 2
        data.append(np.array([*line[:mid], *line[mid+1:], line[mid]]))
        
data = np.array(data)

EMBEDDING_DIM = 200
VOCAB_SIZE = len(vocab)
BATCH_SIZE = 8192
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create model
model = CBOW(VOCAB_SIZE, EMBEDDING_DIM)
model.to(device)


# vectorize data
context = data[:,0:-1]
target = data[:,-1]

loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.train()

for epoch in range(50):
    print(f'{"-"*10} Epoch {epoch} {"-"*10}')
    start = time.time()
    
    avg_loss = 0
    
    # batches
    for i in range(0, context.shape[0], BATCH_SIZE):
    
        optimizer.zero_grad()

        context_t = torch.tensor(context[i:i + BATCH_SIZE]).to(device)
        target_t = torch.tensor(target[i:i + BATCH_SIZE]).to(device)

        # run model
        probs = model(context_t)

        # get loss
        loss = loss_fn(probs, target_t)
        avg_loss += loss.item()

        # update model
        loss.backward()
        optimizer.step()
        
        i += 1
    
    end = time.time()
    print(f'Finished epoch {epoch}, took {end - start}s, average loss is {avg_loss / (context.shape[0] // BATCH_SIZE + 1)}')

