import torch, torch.nn as nn


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True) # sparse is important because only few embeddings will change
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.embedding_dim = embedding_dim
        self.activation = nn.LogSigmoid()
        
        self.init_weights()

    def forward(self, target, context, negative_samples):
        """
            target - bx1 tensor
            context - bx1 tensor
            negative_samples - bxk tensor
        """
        target_emb = self.target_embeddings(target) # ux
        context_emb = self.context_embeddings(context) # vc
        negative_emb = self.context_embeddings(negative_samples) # vk1, vk2, ...

        # log(o(ux . vc))
        y = torch.mul(target_emb, context_emb) # element wise product
        y = torch.sum(y, dim=1) # dot product considering batch
        y = self.activation(y)
        
        # negative sampling
        # sum log(o(-ux . uk))
        y2 = torch.bmm(negative_emb, -target_emb.unsqueeze(2)).squeeze() # squeeze is needed for bmm to do a bxkxn by bxnx1 multiplication, which results in individual elementwise products
        y2 = self.activation(y2)
        y2 = torch.sum(y2, dim=1) # bxk -> bx1

        return (-y - y2).sum() / target.shape[0]

    def init_weights(self):
        irange = 0.5 / self.embedding_dim
        self.target_embeddings.weight.data.uniform_(-irange,irange)
        self.context_embeddings.weight.data.uniform_(-irange,irange)