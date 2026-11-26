"""Q1/Q2: MLP/CNN experiments. Run: python run_q1_q2.py --epochs 15"""
import argparse, json, copy
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from common import *
from models import MLP, SmallCNN


class RandomLabelDataset(Dataset):
    def __init__(self, base, idx, seed=SEED, fraction=0.25):
        self.base=base; self.idx=np.array(idx); targets=np.array(base.targets)[self.idx].copy()
        rng=np.random.default_rng(seed); changed=rng.choice(len(targets),int(fraction*len(targets)),replace=False)
        targets[changed]=rng.integers(0,10,size=len(changed)); self.targets=targets
    def __len__(self): return len(self.idx)
    def __getitem__(self,i): x,_=self.base[self.idx[i]]; return x,int(self.targets[i])

def train_noisy_labels(model, train_loader, val_loader, epochs=15, lr=1e-3):
    model.to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=lr); hist=[]; best=0
    for ep in range(1,epochs+1):
        model.train(); total=hit=0
        for x,y in train_loader:
            x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad(); z=model(x); loss=nn.functional.cross_entropy(z,y); loss.backward(); opt.step(); hit+=(z.argmax(1)==y).sum().item(); total+=len(y)
        model.eval(); vh=vn=0
        with torch.no_grad():
            for x,y in val_loader:
                z=model(x.to(DEVICE)); vh+=(z.argmax(1).cpu()==y).sum().item(); vn+=len(y)
        row={"epoch":ep,"train_acc_on_corrupted_labels":hit/total,"val_acc_on_clean_labels":vh/vn}; hist.append(row)
        print(f"Noisy-label epoch {ep}: noisy-train accuracy={row['train_acc_on_corrupted_labels']:.3f}, clean val={row['val_acc_on_clean_labels']:.3f}")
        if row['val_acc_on_clean_labels']>best: best=row['val_acc_on_clean_labels']; torch.save(model.state_dict(),result_dir('Q1','models')/'q1_memorization_best.pt')
    save_json(result_dir('Q1','metrics')/'q1_memorization_history.json',hist)

def occlude_center(x):
    x=x.clone(); x[:,:,8:24,8:24]=0.0; return x

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=15); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--workers',type=int,default=2); args=ap.parse_args()
    seed_all(); tr,va,te=get_cifar_loaders(args.batch_size,args.workers)
    # Q1(a): capacity/depth comparison. Shared optimizer, epochs, batch size, augmentation and split.
    mlp512=MLP((512,)); mlp256x2=MLP((256,256))
    print('Q1 MLP-512 parameters:',count_params(mlp512)); print('Q1 MLP-256x256 parameters:',count_params(mlp256x2))
    mlp512,h1=train_classifier(mlp512,tr,va,args.epochs,name='q1_mlp512')
    mlp256x2,h2=train_classifier(mlp256x2,tr,va,args.epochs,name='q1_mlp256x256')
    q1_test={'mlp512':evaluate_classifier(mlp512,te),'mlp256x256':evaluate_classifier(mlp256x2,te)}
    save_json(result_dir('Q1','metrics')/'q1_test_metrics.json',q1_test)
    # Q1(b): intentionally corrupted labels on 25% of training examples; track over time.
    base_aug=datasets.CIFAR10(DATA_DIR,train=True,download=False,transform=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor()]))
    base_plain=datasets.CIFAR10(DATA_DIR,train=True,download=False,transform=transforms.ToTensor())
    labels=np.asarray(base_plain.targets); train_idx,val_idx=train_test_split(np.arange(len(labels)),test_size=5000,random_state=SEED,stratify=labels)
    noisy=RandomLabelDataset(base_aug,train_idx,fraction=0.25)
    noisy_loader=DataLoader(noisy,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=torch.cuda.is_available())
    val_loader=DataLoader(Subset(base_plain,val_idx),batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
    train_noisy_labels(MLP((1024,1024)),noisy_loader,val_loader,args.epochs)
    # Q2 clean baseline: train CNN and MLP under same data split/epochs/batch/update budget.
    cnn=SmallCNN(); mlp_q2=MLP((512,))
    print('Q2 CNN params:',count_params(cnn),'MLP params:',count_params(mlp_q2))
    cnn,_=train_classifier(cnn,tr,va,args.epochs,name='q2_cnn_clean')
    mlp_q2,_=train_classifier(mlp_q2,tr,va,args.epochs,name='q2_mlp_clean')
    # Evaluate clean and central-occlusion condition with exact same test samples.
    _,_,test_loader=get_cifar_loaders(args.batch_size,args.workers)
    clean_cnn=evaluate_classifier(cnn,test_loader); occ_cnn=evaluate_classifier(cnn,test_loader,occlude_center)
    clean_mlp=evaluate_classifier(mlp_q2,test_loader); occ_mlp=evaluate_classifier(mlp_q2,test_loader,occlude_center)
    save_json(result_dir('Q2','metrics')/'q2_occlusion_results.json',{'cnn_clean':clean_cnn,'cnn_center_occlusion':occ_cnn,'mlp_clean':clean_mlp,'mlp_center_occlusion':occ_mlp})
    # Q2(b): retrain both after fixed pixel-location permutation.
    trs,vas,tes=get_cifar_loaders(args.batch_size,args.workers,spatial_shuffle=True)
    cnn_sh=SmallCNN(); mlp_sh=MLP((512,))
    cnn_sh,_=train_classifier(cnn_sh,trs,vas,args.epochs,name='q2_cnn_spatial_shuffle')
    mlp_sh,_=train_classifier(mlp_sh,trs,vas,args.epochs,name='q2_mlp_spatial_shuffle')
    results={'clean_cnn':clean_cnn,'clean_mlp':clean_mlp,'occluded_cnn':occ_cnn,'occluded_mlp':occ_mlp,
             'spatial_shuffle_cnn':evaluate_classifier(cnn_sh,tes),'spatial_shuffle_mlp':evaluate_classifier(mlp_sh,tes)}
    save_json(result_dir('Q2','metrics')/'q1_q2_summary.json',results)
    print('Q1/Q2 runs complete. Outputs are organized under results/Q1 and results/Q2.')
if __name__=='__main__': main()
