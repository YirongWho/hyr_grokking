import math
from argparse import ArgumentParser
from itertools import permutations

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_tokens, embedding_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.num_tokens = num_tokens
        self.embedding = nn.Embedding(num_tokens, embedding_dim)
        self.fc1 = nn.Linear(4*embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape (4, k)

        # Pass each part through the embedding layer
        x_parts = [self.embedding(x_part) for x_part in x]
    
        # Concatenate the embeddings
        x = torch.cat(x_parts, dim=1)
    
        # Pass the embeddings through the MLP
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


def Addition_mod_p_data(p, eq_token, op_token):
    """
    x+y
    """
    x = torch.arange(p)
    y = torch.arange(1,p)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    result = (x + y) % p

    # our experiments use a 3 layer MLP with an embedding trained on datasets of
    # equations of the form a◦b = c, where each of “a”, “◦”, “b”, “=”, and “c”
    # is a seperate token"
    return torch.stack([x, op, y, eq, result])


def main(args):
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokens for <op> and <=>. It's not clear why <=> is needed at all since it
    # has no effect on the output, but we'll leave it in to best follow the
    # paper.
    eq_token = args.p
    op_token = args.p + 1

    # We trained a MLP
    
    model=MLP(num_tokens=args.p+2, embedding_dim=128,hidden_dim=512, output_dim=args.p+2).to(device)

    # "We train on the binary operation of Addition mod 97 with 50% of the data
    # in the training set."
    data = Addition_mod_p_data(args.p, eq_token, op_token)
    train_fraction = args.train_fraction
    train_size = int(train_fraction * data.shape[1])
    rand_perm=torch.randperm(data.shape[1])
    train_idx, valid_idx = rand_perm[:train_size], rand_perm[train_size:]
    train_data, valid_data = data[:, train_idx], data[:, valid_idx]


    # For most experiments we used AdamW optimizer with learning rate 10−3,
    # weight decay 1, β1 = 0.9, β2 = 0.98
    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )

    #  linear learning rate warmup over the first 10 updates
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: 1 if update > 10 else update / 10
    )

    steps_per_epoch = math.ceil(train_data.shape[1] / args.batch_size)

    train_acc, val_acc, train_loss, val_loss = [], [], [], []

    for e in tqdm(range(int(args.budget) // steps_per_epoch)):

        # randomly shuffle train data
        train_data = train_data[:, torch.randperm(train_data.shape[1])]

        for data, is_train in [(train_data, True), (valid_data, False)]:

            model.train(is_train)
            total_loss = 0
            total_acc = 0

            # torch.split faster than dataloader with tensor
            dl = torch.split(data, args.batch_size, dim=1)
            for input in dl:
                input = input.to(device)

                with torch.set_grad_enabled(is_train):
                    logits = model(input[:-1])
                    # calculate loss only on the answer part of the equation (last element
                    loss = F.cross_entropy(logits, input[-1])
                    total_loss += loss.item() * input.shape[-1]

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                acc = (logits.argmax(-1) == input[-1]).float().mean()
                total_acc += acc.item() * input.shape[-1]

            if is_train:
                train_acc.append(total_acc / train_data.shape[-1])
                train_loss.append(total_loss / train_data.shape[-1])
            else:
                val_acc.append(total_acc / valid_data.shape[-1])
                val_loss.append(total_loss / valid_data.shape[-1])

        if (e + 1) % 100 == 0:
            steps = torch.arange(len(train_acc)).numpy() * steps_per_epoch
            # save steps, train_acc, val_acc, train_loss, val_loss to a file
            torch.save((steps, train_acc, val_acc, train_loss, val_loss), f"datas/mlp/frac={args.train_fraction}.pt")
            torch.save(model.state_dict(), f"results/mlp_{args.train_fraction}.pth")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--p", type=int, default=23)
    parser.add_argument("--budget", type=int, default=1e5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--train_fraction", type=float, default=0.5)  
    args = parser.parse_args()
    main(args)
