import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import IMDB
from torch.nn.utils.rnn import pad_sequence

# Subword n-gram generator for FastText
def get_subwords(word, n=3):
    return [word[i:j] for i in range(len(word)) for j in range(i+1, min(i+n+1, len(word)+1))]

# FastText Embedding Model
class FastTextEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.EmbeddingBag(vocab_size, embed_dim)
    
    def forward(self, subword_indices):
        return self.embed(subword_indices)

# Data Preparation
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

def collate_batch(batch):
    label_list, text_list = [], []
    for _label, _text in batch:
        subwords = []
        for word in tokenizer(_text):
            subwords.extend(get_subwords(word))
        text_list.append(torch.tensor(vocab(subwords), dtype=torch.long))
        label_list.append(torch.tensor(_label-1))  # Adjust labels (1,2) to (0,1)
    label_list = torch.stack(label_list)
    text_list = pad_sequence(text_list, batch_first=True)
    return label_list.to(device), text_list.to(device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train FastText (simplified)
vocab_size = len(vocab)
embed_dim = 100
ft_model = FastTextEmbedding(vocab_size, embed_dim).to(device)
optimizer = optim.Adam(ft_model.parameters(), lr=0.001)

class EmbeddingTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
    
    def train_step(self, inputs):
        self.model.zero_grad()
        embeds = self.model(inputs)
        loss = nn.MSELoss()(embeds.mean(0), torch.zeros_like(embeds.mean(0)))  # Simplified CBOW-like
        loss.backward()
        self.optimizer.step()
        return loss.item()

trainer = EmbeddingTrainer(ft_model, optimizer)
# Example training (subset for demo)
sample_batch = next(iter(IMDB(split='train')))
labels, texts = collate_batch(sample_batch)
loss = trainer.train_step(texts)
print(f"Sample Loss: {loss}")