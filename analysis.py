import torch
from matplotlib import pyplot as plt

def main(args):
    model_name=args.model_name
    frac=args.train_fraction
    file_path = f"curve_data/{model_name}/frac={frac}.pt"
    # read .pt file 
    steps, train_acc, val_acc, train_loss, val_loss = torch.load(file_path)

    # print the maximal of val_acc
    print(f"maximal of val_acc: {max(val_acc)}")

    # caculate the step when val_acc reached 0.95 at first time
    for i, acc in enumerate(val_acc):
        if acc >= 0.95:
            print(f"val_acc reached 0.9 at step {steps[i]}")
            break
    if args.v==True:
        plt.plot(steps, train_acc, label="train")
        plt.plot(steps, val_acc, label="val")
        plt.legend()
        plt.title(f"Modular Addition (training on {int(args.train_fraction*100)}% of data)")
        plt.xlabel("Optimization Steps")
        plt.ylabel("Accuracy")
        plt.xscale("log", base=10)
        plt.savefig(f"figures/{model_name}/{model_name}_acc_frac={frac}.png", dpi=150)
        plt.close()

        plt.plot(steps, train_loss, label="train")
        plt.plot(steps, val_loss, label="val")
        plt.legend()
        plt.title(f"Modular Addition (training on {int(args.train_fraction*100)}% of data)")
        plt.xlabel("Optimization Steps")
        plt.ylabel("Loss")
        plt.xscale("log", base=10)
        plt.savefig(f"figures/{model_name}/{model_name}_loss_frac={frac}.png", dpi=150)
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mlp")
    parser.add_argument("--train_fraction", type=float, default=0.5)
    parser.add_argument("--v",type=bool,default=False)
    args = parser.parse_args()
    main(args)