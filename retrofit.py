#Faruqui, Manaal, et al. "Retrofitting word vectors to semantic lexicons."

import sys
from word2vec.model import SkipGram
import torch
from nltk.corpus import wordnet as wn

from copy import deepcopy

def generate_lexicon(vocab):
    """
        Gets the synonyms of each word in the vocabulary
        returns it as dictionary
    """

    return {vocab_word: set([word.strip().lower() for syn in wn.synsets(vocab_word) for word in syn.lemma_names() if word.strip().lower() in vocab]) for vocab_word in vocab}


def retrofit(embeddings, lexicon, iters):
    new_embeddings = deepcopy(embeddings)

    for _ in range(iters):
        for word, neighbors in lexicon.items():
            n_neighbors = len(neighbors)

            if n_neighbors == 0:
                continue

            new_embedding = n_neighbors * embeddings[word]

            for new_word in neighbors:
                new_embedding += new_embeddings[new_word]
            new_embeddings[word] = new_embedding / (2 * n_neighbors)
    
    return new_embeddings

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(f'Usage: python3 {sys.argv[0]} <model path> <vocab path> <num_iters>')
        sys.exit(0)
    model_path = sys.argv[1]
    iters = int(sys.argv[3])

    # load vocab
    with open(f'{sys.argv[2]}') as f:
        vocab = sorted(f.readline().split())
    VOCAB_SIZE = len(vocab)

    # String2int conversion
    words_to_idx = {i:j for j,i in enumerate(vocab)}
    idx_to_words = {i:j for i,j in enumerate(vocab)}
    vocab = set(vocab) # performance increase for generate_lexicon

    # get embeddings
    emb_size = int(model_path.split('/')[-1].split('_')[0])
    model = SkipGram(VOCAB_SIZE, emb_size)
    model.load_state_dict(torch.load(model_path))

    embeddings = model.target_embeddings.weight.data.numpy()

    # {word : set(synonims)}
    lexicon = generate_lexicon(vocab)
    lexicon = {words_to_idx[word] : set([words_to_idx[v] for v in values]) for word, values in lexicon.items()}
    
    result_embeddings = retrofit(embeddings, lexicon, iters)

