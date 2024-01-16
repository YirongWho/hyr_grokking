#%%
import torch
maximal_val_acc=[]
fractions=[0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75]
reach=[3e4]*len(fractions)
for fraction in fractions:
    # read from file
    steps, train_acc, val_acc, train_loss, val_loss = torch.load(f"curve_data/mlp/frac={fraction}.pt")
    maximal_val_acc.append(max(val_acc))
    # print the maximal of val_acc
    print(f"maximal of val_acc: {max(val_acc)}")
    # caculate the step when val_acc reached 0.9 at first time
    for i, acc in enumerate(val_acc):
        if acc >= 0.9:
            reach[fractions.index(fraction)]=steps[i]
            break

import matplotlib.pyplot as plt
plt.plot(fractions, maximal_val_acc)
plt.title("Maximal Validation Accuracy")
plt.xlabel("Training Fraction")
plt.ylabel("Accuracy")
plt.savefig("figures/mlp/mlp_maximal_val_acc.png", dpi=150)
plt.close()

colors = []
markers = []

f1=[]
f2=[]
r1=[]
r2=[]
for i in range(len(fractions)):
    if reach[i]<3e4:
        f1.append(fractions[i])
        r1.append(reach[i])
    else:
        f2.append(fractions[i])
        r2.append(reach[i])

plt.scatter(f2, r2, c="r", marker="p", label=f"Fail to  reach 90% in {r'$10^5$'} steps.")  
plt.scatter(f1, r1, c="g", marker="o", label=f"Reach 90% in {r'$10^5$'} steps.")
plt.plot(fractions, reach, 'b-')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.title("Steps until generalization for modular addition")
plt.xlabel("Training data Fraction")
plt.ylabel("Optimization Steps to validation accuracy >90%") 
plt.legend()
plt.savefig("figures/mlp/mlp_steps_until_generalization.png", dpi=150)
plt.close()

#%%
#%%
import torch
maximal_val_acc=[]
fractions=[0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75]
reach=[3e5]*len(fractions)
for fraction in fractions:
    # read from file
    steps, train_acc, val_acc, train_loss, val_loss = torch.load(f"curve_data/lstm/frac={fraction}.pt")
    maximal_val_acc.append(max(val_acc))
    # print the maximal of val_acc
    print(f"maximal of val_acc: {max(val_acc)}")
    # caculate the step when val_acc reached 0.9 at first time
    for i, acc in enumerate(val_acc):
        if acc >= 0.95:
            reach[fractions.index(fraction)]=steps[i]
            break

import matplotlib.pyplot as plt
plt.plot(fractions, maximal_val_acc)
plt.title("Maximal Validation Accuracy")
plt.xlabel("Training Fraction")
plt.ylabel("Accuracy")
plt.savefig("figures/lstm/lstm_maximal_val_acc.png", dpi=150)
plt.close()

colors = []
markers = []

f1=[]
f2=[]
r1=[]
r2=[]
for i in range(len(fractions)):
    if reach[i]<3e5:
        f1.append(fractions[i])
        r1.append(reach[i])
    else:
        f2.append(fractions[i])
        r2.append(reach[i])

plt.scatter(f2, r2, c="r", marker="p", label=f"Fail to  reach 95% in {r'$10^5$'} steps.")  
plt.scatter(f1, r1, c="g", marker="o", label=f"Reach 95% in {r'$10^5$'} steps.")
plt.plot(fractions, reach, 'b-')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.title("Steps until generalization for modular addition")
plt.xlabel("Training data Fraction")
plt.ylabel("Optimization Steps to validation accuracy >90%") 
plt.legend()
plt.savefig("figures/lstm/lstm_steps_until_generalization.png", dpi=150)
plt.close()
# %%
