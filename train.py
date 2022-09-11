import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import string

class PoemsSet(Dataset):

    def __init__(self, 
                 path : str, 
                 seq_length : int):
        """
            DataSet of Egor Letov's poems for RNN model
            path - path to poems' list
            seq_length - length of sequencies in text
        """

        self.seq_length = seq_length

        self.dataset = []
        with open(path, "r") as file:
            raw_text = "".join(line for line in file)
            for cur_poem in raw_text.split("-----\n"):
                self.dataset.extend(self.poem_to_seq(cur_poem))

        self.unique_words = np.unique(self.dataset)

        self.words_to_idxs = {word : idx for idx, word in enumerate(self.unique_words)}
        self.idxs_to_words = {idx : word for idx, word in enumerate(self.unique_words)}

        self.idxs = [self.words_to_idxs[word] for word in self.dataset]

    def poem_to_seq(self, poem : str) -> list:
        """
            Gets raw poem, parse it to standart form
        """

        cutoff_date = poem[:poem.rfind("\n\n")]
        lower_text = " ".join(cutoff_date.split("\n")).lower()

        return "".join([c for c in lower_text if c not in string.punctuation]).split()

    def __len__(self):
        return len(self.idxs) - self.seq_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.idxs[index : index + self.seq_length]),
            torch.tensor(self.idxs[index + 1: index + self.seq_length + 1])
        )


class RNN_model(nn.Module):

    def __init__(self, 
                 dataset : Dataset):
        """
            RNN model that will use for generating text
        """
        super().__init__()

        self.unique_words = len(dataset.unique_words)
        self.embedding_dim = 256
        self.gru_inp = 256
        self.gru_layers = 3

        self.embedding = nn.Embedding(
            num_embeddings=self.unique_words,
            embedding_dim=self.embedding_dim
        )

        self.gru = nn.GRU(
            input_size=self.gru_inp,
            hidden_size=self.gru_inp,
            num_layers=self.gru_layers,
            dropout=0.3
        )
        self.linear = nn.Linear(self.gru_inp, self.unique_words)

    def forward(self, x, context):
        emb = self.embedding(x)
        output, new_context = self.gru(emb, context)
        logits = self.linear(output)

        return logits, new_context
    
    def init_state(self, seq_length : int):
        return torch.zeros(self.gru_layers, seq_length, self.gru_inp)


class ModelTrainer:
    
    def __init__(self):
        """
            Creates and trains model, that should generate text
        """

        pass

    def fit(self, 
            path : str,     
            seq_length : int = 12,
            num_epochs : int = 3,
            lr : float = 3e-4, 
            batch_size=512):
        """
            Fits the model 
            path - path to .txt file where poems 
            separates with "-----"
            seq_length - length of sequence in train            
        """

        self.dataset = PoemSet(path=path, seq_lengt=seq_length)
        dataloader = DataLoader(self.dataset, batch_size=batch_size)

        rnn_model = RNN_model(dataset=self.dataset)

        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(rnn_model.parameters(), lr=lr)

        model.train()

        for epoch in range(num_epochs):
            h_cur = model.init_state(seq_length)
            cur_loss = []
            
            for batch_id, (X, y) in enumerate(dataloader):
                
                pred, h_new = model(X, h_cur)
                loss = criterion(pred.transpose(1, 2), y)

                h_cur = h_new.detach()

                loss.backward()
                opt.step()
                opt.zero_grad()

                cur_loss.append(loss.item())
        
                if batch_id % 10 == 0:
                    print(f"Batch num {batch_id}! Cur loss: {cur_loss[-1]}")

            print(f"Epoch {epoch}, loss = {np.mean(cur_loss)}")

        self.model = rnn_model

    def generate(self, 
                 prefix : str = None,
                 num_words : int = 15) -> str:
        """
            generates words from trained model
            prefix - word for start of generating
            num_words - number of words to generate 
        """        
            
        if prefix is None:
            prefix = np.random.choice(self.dataset.dataset)

        words = prefix.split()
        h_cur = model.init_state(len(words))

        generated = words

        for word in range(num_gen):
            X = torch.tensor([[dataset.words_to_idxs[word] for word in words]])
            pred, h_cur = model(X, h_cur)
            lastword_logits = pred[0][-1]
            p = torch.nn.functional.softmax(lastword_logits, dim=0).detach().numpy()
            next_idx = np.random.choice(len(lastword_logits), p=p)
            generated.append(dataset.idxs_to_words[next_idx])

        return " ".join(generated)


if __name__ == "__main__":
    print("Hello")














