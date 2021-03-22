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


class WikiDataset(torch.utils.data.IterableDataset):
    def __init__(self, file):        
        self.file = file

    def __iter__(self):
        return WikiData(self.file)

class WikiData:
    def __init__(self, filename):
        self.file = open(filename)

    def __next__(self):
        line = tuple(map(lambda x: words_to_idx[x], next(self.file).split()))
        mid = len(line) // 2
        return np.array([*line[:mid], *line[mid+1:]]), np.array(line[mid])

EMBEDDING_DIM = 200
BATCH_SIZE = 8192
NUM_WORKERS = 2
device = "cuda" if torch.cuda.is_available() else "cpu"

filename = 'tmp.txt'

with open(f'{filename}.vocab') as f:
    vocab = sorted(f.readline().split())
VOCAB_SIZE = len(vocab)

# data = []
words_to_idx = {i:j for j,i in enumerate(vocab)}
        
# data = np.array(data)
train = WikiDataset(f'{filename}.examples.train')
dev = WikiDataset(f'{filename}.examples.dev')
test = WikiDataset(f'{filename}.examples.test')

train_dataloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
dev_dataloader = torch.utils.data.DataLoader(dev, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
test_dataloader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# Create model
model = CBOW(VOCAB_SIZE, EMBEDDING_DIM)
model.to(device)

loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

def test(dataloader):
  model.eval()
  avg_loss = 0
  correct = 0
  count = 0
  for i, (context_t, target_t) in enumerate(dataloader):
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

  print(f'Testing average loss is {avg_loss / i}, accuracy is {correct / count}')

  def train():
  for epoch in range(50): 
      model.train()
      
      print(f'{"-"*10} Epoch {epoch} {"-"*10}')
      start = time.time()
      avg_loss = 0
      correct = 0
      count = 0
      # batches
      for i, (context_t, target_t) in enumerate(train_dataloader):
      
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
      test(dev_dataloader)

train()