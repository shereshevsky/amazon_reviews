import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import VOCAB_SIZE, EMBEDDING_SIZE


class CNN_Text(nn.Module):

    def __init__(self, n_classes, embedding_matrix):
        super(CNN_Text, self).__init__()
        filter_sizes = [1, 2, 3, 5]
        num_filters = 36
        n_classes = n_classes
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, EMBEDDING_SIZE)) for K in filter_sizes])
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc1(x)
        return logit
