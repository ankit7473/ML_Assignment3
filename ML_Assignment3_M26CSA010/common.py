"""Common utilities for ML Assignment 3 (CIFAR-10)."""
from pathlib import Path
import os, random, json, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

SEED = 26010
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

def result_dir(question, kind=None):
    path = OUT / question
    if kind: path = path / kind
    path.mkdir(parents=True, exist_ok=True)
    return path

def result_path(name, kind='metrics'):
    q = name.split('_', 1)[0].upper()
    if q not in {'Q1','Q2','Q3','Q4'}: q = 'Q1'
    return result_dir(q, kind)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cifar_loaders(batch_size=128, workers=2, spatial_shuffle=False, train_subset=None):
    """One fixed stratified 45k/5k split. Only training transforms are stochastic."""
    seed_all(SEED)
    train_aug = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    plain = transforms.ToTensor()
    base = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=plain)
    y = np.asarray(base.targets)
    train_idx, val_idx = train_test_split(np.arange(len(y)), test_size=5000, random_state=SEED, stratify=y)
    train_ds = datasets.CIFAR10(DATA_DIR, train=True, download=False, transform=train_aug)
    val_ds = datasets.CIFAR10(DATA_DIR, train=True, download=False, transform=plain)
    test_ds = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=plain)
    if train_subset:
        rng = np.random.default_rng(SEED)
        train_idx = rng.choice(train_idx, train_subset, replace=False)
    if spatial_shuffle:
        perm = np.random.default_rng(SEED).permutation(32*32)
        class SpatialShuffle(torch.utils.data.Dataset):
            def __init__(self, ds): self.ds=ds
            def __len__(self): return len(self.ds)
            def __getitem__(self, i):
                x,y=self.ds[i]; x=x.reshape(3,-1)[:,perm].reshape(3,32,32); return x,y
        train_ds, val_ds, test_ds = SpatialShuffle(train_ds), SpatialShuffle(val_ds), SpatialShuffle(test_ds)
    kwargs = dict(batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available())
    return (DataLoader(Subset(train_ds, train_idx), shuffle=True, **kwargs),
            DataLoader(Subset(val_ds, val_idx), shuffle=False, **kwargs),
            DataLoader(test_ds, shuffle=False, **kwargs))

def count_params(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_classifier(model, train_loader, val_loader, epochs=15, lr=1e-3, name="model"):
    model=model.to(DEVICE); opt=torch.optim.Adam(model.parameters(), lr=lr); loss_fn=nn.CrossEntropyLoss()
    best_val=float("inf"); history=[]; start=time.time()
    for ep in range(1,epochs+1):
        model.train(); train_loss=0; train_correct=0; train_n=0
        for x,y in train_loader:
            x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad(); logits=model(x); loss=loss_fn(logits,y); loss.backward(); opt.step()
            train_loss+=loss.item()*len(y); train_correct+=(logits.argmax(1)==y).sum().item(); train_n+=len(y)
        model.eval(); val_loss=0; val_correct=0; val_n=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y=x.to(DEVICE),y.to(DEVICE); logits=model(x); val_loss+=loss_fn(logits,y).item()*len(y); val_correct+=(logits.argmax(1)==y).sum().item(); val_n+=len(y)
        row={"epoch":ep,"train_loss":train_loss/train_n,"train_acc":train_correct/train_n,"val_loss":val_loss/val_n,"val_acc":val_correct/val_n}
        history.append(row); print(f"{name} ep {ep}/{epochs}: train acc {row['train_acc']:.3f} val acc {row['val_acc']:.3f}")
        if row["val_loss"]<best_val:
            best_val=row["val_loss"]; torch.save(model.state_dict(), result_path(name, 'models')/f"{name}_best.pt")
    torch.save(model.state_dict(), result_path(name, 'models')/f"{name}_last.pt")
    model.load_state_dict(torch.load(result_path(name, 'models')/f"{name}_best.pt", map_location=DEVICE))
    with open(result_path(name, 'metrics')/f"{name}_history.json","w") as f: json.dump(history,f,indent=2)
    print(f"Finished {name} in {time.time()-start:.1f}s; best val loss {best_val:.4f}")
    return model,history

def evaluate_classifier(model, loader, corrupt=None):
    model.eval(); correct=0; n=0; loss=0; confidence=[]; preds=[]; labels=[]
    with torch.no_grad():
        for x,y in loader:
            if corrupt is not None: x=corrupt(x)
            x,y=x.to(DEVICE),y.to(DEVICE); logits=model(x); probs=logits.softmax(1)
            loss+=nn.functional.cross_entropy(logits,y,reduction="sum").item(); correct+=(logits.argmax(1)==y).sum().item(); n+=len(y)
            confidence.extend(probs.max(1).values.cpu().tolist()); preds.extend(logits.argmax(1).cpu().tolist()); labels.extend(y.cpu().tolist())
    return {"accuracy":correct/n,"loss":loss/n,"mean_confidence":float(np.mean(confidence)),"predictions":preds,"labels":labels}

def save_json(path, data):
    with open(path,"w") as f: json.dump(data,f,indent=2)
