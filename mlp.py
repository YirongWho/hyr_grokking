import math
from argparse import ArgumentParser
from itertools import permutations

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import copy

class MLP(nn.Module):
    def __init__(self, num_tokens, embedding_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.num_tokens = num_tokens
        self.embedding = nn.Embedding(num_tokens, embedding_dim)
        #self.embedding.weight.requires_grad = False
        self.fc1 = nn.Linear(2*embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape (2, k)

        # Pass each part through the embedding layer
        x_parts = [self.embedding(x_part) for x_part in x]
    
        # Concatenate the embeddings
        x = torch.cat(x_parts, dim=1)
    
        # Pass the embeddings through the MLP
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def Group_data(gp_struct):
    '''
    gp_sturct is a list : [p1,p2,...,pk]. Then the group=Z/p1*Z/p2*...*Z/pk
    '''
    q = 1
    for p in gp_struct:
        q *= p
    # construct a list of tensor
    x = [torch.arange(p) for p in gp_struct]
    # cartesian product
    x = torch.cartesian_prod(*x) # x.shape=(q,k)
    gp_data = torch.zeros((3, q*(q-1)), dtype=torch.int64) # Specify dtype as torch.int64
    for i in range(q):
        for j in range(1, q): # j=0 is the identity element, used only once.
            factor = 1
            for k in range(len(gp_struct)):
                gp_data[0, i*(q-1)+j-1] += x[i, k]*factor
                gp_data[1, i*(q-1)+j-1] += x[j, k]*factor
                gp_data[2, i*(q-1)+j-1] += ((x[i, k]+x[j, k]) % gp_struct[k])*factor
                factor *= gp_struct[k]
    return gp_data
            

def Addition_mod_q_data(q, eq_token, op_token):
    """
    x+y
    """
    p = int(math.sqrt(q))
    x = torch.arange(q)
    y = torch.arange(1,q)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    # x=a*p+b, y=c*p+d
    a = x//p
    b = x%p
    c = y//p
    d =  y%p
    result = ((a+c)%p)*p+(b+d)%p

    # our experiments use a 3 layer MLP with an embedding trained on datasets of
    # equations of the form a◦b = c, where each of “a”, “◦”, “b”, “=”, and “c”
    # is a seperate token"
    return torch.stack([x, op, y, eq, result])

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
    torch.manual_seed(42)
    gp_size=1
    for p in args.group:
        gp_size *= p
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We trained a MLP
    model=MLP(num_tokens=gp_size, embedding_dim=args.embedding,hidden_dim=args.MLP_width, output_dim=gp_size).to(device)
#     init_weight = copy.deepcopy(model.embedding.weight)
    #param = torch.load('params/mlp_8_32.pth')
    # model.load_state_dict(param)
#     init_weight_norm = torch.norm(init_weight,dim=1).reshape(-1,1)
#     init_weight = init_weight/init_weight_norm
#     model.embedding.weight = nn.Parameter(init_weight.to(device))

    # model.fc3.weight = nn.Parameter(param['fc3.weight'].to(device))
    # model.fc3.bias = nn.Parameter(param['fc3.bias'].to(device))

    # "We train on the binary operation of Addition mod 97 with 50% of the data
    # in the training set."
    data = Group_data(args.group)
    train_idx, valid_idx = torch.randperm(data.shape[1]).split(data.shape[1] // 2)
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

    flag_train = 0
    flag_val = 0
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
                if train_acc[-1]>0.99  and flag_train==0:
                    print(f'train acc=1 steps:{len(train_acc)*steps_per_epoch}')
                    flag_train = 1
            else:
                val_acc.append(total_acc / valid_data.shape[-1])
                val_loss.append(total_loss / valid_data.shape[-1])
                if val_acc[-1]>0.99  and flag_val==0:
                    print(f'val acc=1 steps:{len(train_acc)*steps_per_epoch}')
                    flag_val = 1

        if (e + 1) % 1000 == 0:
            steps = torch.arange(len(train_acc)).numpy() * steps_per_epoch
            plt.plot(steps, train_acc, label="train")
            plt.plot(steps, val_acc, label="val")
            plt.legend()
            plt.title(f"Abelian group {args.group} (training on 50% of data)")
            plt.xlabel("Optimization Steps")
            plt.ylabel("Accuracy")
            plt.xscale("log", base=10)
            plt.savefig(f"figures/{args.filename}_acc.png", dpi=150)
            plt.close()

            plt.plot(steps, train_loss, label="train")
            plt.plot(steps, val_loss, label="val")
            plt.legend()
            plt.title(f"Abelian group {args.group} (training on 50% of data)")
            plt.xlabel("Optimization Steps")
            plt.ylabel("Loss")
            plt.xscale("log", base=10)
            plt.savefig(f"figures/{args.filename}_loss.png", dpi=150)
            plt.close()
            torch.save(model.state_dict(),f'params/{args.filename}.pth')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--group", type=lambda x: [int(i) for i in x.split(',')], default=[2,3,7])
    parser.add_argument("--embedding", type=int, default=8)
    parser.add_argument("--MLP_width", type=int, default=32)
    parser.add_argument("--budget", type=int, default=3e5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--filename",default="new_experiment")
    args = parser.parse_args()
    main(args)
