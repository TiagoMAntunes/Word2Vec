import torch
from word2vec.model import SkipGram
from sklearn.metrics.pairwise import cosine_similarity

# load vocab
with open(f'word2vec/new_wiki.txt.vocab') as f:
    vocab = sorted(f.readline().split())
VOCAB_SIZE = len(vocab)

# String2int conversion
words_to_idx = {i:j for j,i in enumerate(vocab)}
idx_to_words = {i:j for i,j in enumerate(vocab)}

model = SkipGram(VOCAB_SIZE,300)
model.load_state_dict(torch.load('300_skipgram', map_location="cpu"))
embeddings = model.target_embeddings.weight.data.numpy()

words = list(map(lambda x: embeddings[words_to_idx[x]], ['tokyo','japan','paris']))
input_words = ['tokyo','japan','paris']

values = []
for word in vocab:
    if word in input_words:
            continue
    test = cosine_similarity((words[1] - words[0]).reshape(1,-1), (embeddings[words_to_idx[word]] - words[2]).reshape(1,-1))[0][0]
    values.append((word, test))

values.sort(key=lambda x: -x[1])
print(values[:10])
('nippon', 0.42539185), ('japans', 0.28567222), ('tempera', 0.24193455), ('britain', 0.23501676), ('japanning', 0.22885758), 