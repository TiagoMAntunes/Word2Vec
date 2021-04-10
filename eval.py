# this is the script that will evaluate the performance of a given model
import sys
import torch
from word2vec.model import SkipGram
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

TEST_FILE_PATH = 'wordsim353/combined.tab'

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'Usage: python3 {sys.argv[0]} <vocab> <model path>+')
        sys.exit(0)

    for index, filename in enumerate(sys.argv[2:]):
        print(f'Model {index + 1}: {filename}')


    # load vocab
    with open(f'{sys.argv[1]}.vocab') as f:
        vocab = sorted(f.readline().split())
    VOCAB_SIZE = len(vocab)

    # String2int conversion
    words_to_idx = {i:j for j,i in enumerate(vocab)}
    idx_to_words = {i:j for i,j in enumerate(vocab)}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = []
    for name in sys.argv[2:]:
        emb_size = int(name.split('/')[-1].split('_')[0])
        models.append(SkipGram(VOCAB_SIZE, emb_size))
        models[-1].load_state_dict(torch.load(name, map_location=device))
        models[-1].eval()


    with open(TEST_FILE_PATH) as f:
        next(f) # skip first line
        
        # [word1, word2, val]
        lines = list(
                    map(lambda x: [words_to_idx[x[0]], words_to_idx[x[1]], x[2]],
                    filter(lambda x: x[0] in vocab and x[1] in vocab, 
                    map(lambda x: [x[0].lower(), x[1].lower(), float(x[2])], 
                    map(lambda x: x.split(), f.readlines()
        )))))

    for i in range(len(models)):
        print('-'*5, sys.argv[2:][i], '-'*5)
        emb = models[i].target_embeddings
        with open(f'{sys.argv[2:][i].split("/")[-1]}.out', 'w+') as f:
            f.write(f'Word1,Word2,Result,Base Result\n')
            for line in lines:
                a = emb(torch.tensor(line[0])).detach().reshape(1, -1)
                b = emb(torch.tensor(line[1])).detach().reshape(1, -1)
                f.write(f'{idx_to_words[line[0]]},{idx_to_words[line[1]]},{cosine_similarity(a,b)[0][0]},{(line[2]-5)/5}\n')
            

