import torch, torch.nn as nn
from dataloader import WikiDataset
from model import SkipGram
import time

def train(model, dataloader, optimizer, epochs=50, k = 5):
    for epoch in range(epochs):
        model.train()

        print(f'{"-"*10} Epoch {epoch} {"-"*10}')
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
            # FIXME
            negative_samples = None

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


if __name__ == '__main__':
    EMBEDDING_DIM = 200
    BATCH_SIZE = 4096
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

    model = SkipGram(VOCAB_SIZE, EMBEDDING_DIM)
    model.to(DEVICE)
    model.device = DEVICE

    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

    train(model, train_dataloader, optimizer)

