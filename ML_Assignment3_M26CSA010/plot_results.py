"""Generate assignment deliverable figures from completed result JSON files."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent; R=ROOT/'results'

def read(path):
    return json.load(open(path)) if path.exists() else None

def plot_history(folder, filename, title=None):
    h=read(folder/'metrics'/filename)
    if not h: return
    epochs=[r['epoch'] for r in h]
    plt.figure(); plt.plot(epochs,[r['train_acc'] for r in h],label='Train'); plt.plot(epochs,[r['val_acc'] for r in h],label='Validation')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title(title or filename.replace('_history.json','')); plt.legend(); plt.tight_layout()
    out=folder/'plots'; out.mkdir(exist_ok=True); plt.savefig(out/filename.replace('_history.json','_accuracy.png'),dpi=200); plt.close()

# Q1/Q2 accuracy curves
for folder,files in [(R/'Q1',['q1_mlp512_history.json','q1_mlp256x256_history.json']), (R/'Q2',['q2_cnn_clean_history.json','q2_mlp_clean_history.json','q2_cnn_spatial_shuffle_history.json','q2_mlp_spatial_shuffle_history.json'])]:
    for f in files: plot_history(folder,f)
# Q1 noisy-label trajectory
q1=read(R/'Q1'/'metrics'/'q1_memorization_history.json')
if q1:
    p=R/'Q1'/'plots'; p.mkdir(exist_ok=True); ep=[x['epoch'] for x in q1]
    plt.figure(); plt.plot(ep,[x['train_acc_on_corrupted_labels'] for x in q1],label='Train accuracy (corrupted labels)'); plt.plot(ep,[x['val_acc_on_clean_labels'] for x in q1],label='Clean validation accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Q1 Memorization trajectory'); plt.legend(); plt.tight_layout(); plt.savefig(p/'q1_memorization.png',dpi=200); plt.close()
# Q3 bottleneck selection
q3=read(R/'Q3'/'metrics'/'q3_capacity_validation.json')
if q3:
    caps=list(q3['validation']); vals=[q3['validation'][k]['pixel_mse'] for k in caps]
    p=R/'Q3'/'plots'; p.mkdir(exist_ok=True); plt.figure(); plt.plot(caps,vals,marker='o'); plt.xlabel('Bottleneck channels'); plt.ylabel('Validation pixel MSE'); plt.title('Q3 Autoencoder capacity selection'); plt.tight_layout(); plt.savefig(p/'q3_capacity_validation.png',dpi=200); plt.close()
# Q4 SVM sweep: validation accuracy and support-vector fraction vs C
q4=read(R/'Q4'/'metrics'/'q4_svm_sweep.json')
if q4:
    p=R/'Q4'/'plots'; p.mkdir(exist_ok=True)
    for metric, ylabel, filename in [('validation_accuracy','Validation accuracy','q4_validation_accuracy.png'),('support_vector_fraction','Training support vector fraction','q4_support_vector_fraction.png')]:
        plt.figure()
        for kernel in sorted({r['kernel'] for r in q4['sweep']}):
            rows=sorted([r for r in q4['sweep'] if r['kernel']==kernel],key=lambda r:r['C'])
            plt.plot([r['C'] for r in rows],[r[metric] for r in rows],marker='o',label=kernel)
        plt.xscale('log'); plt.xlabel('C'); plt.ylabel(ylabel); plt.title('Q4 SVM '+ylabel+' vs C'); plt.legend(); plt.tight_layout(); plt.savefig(p/filename,dpi=200); plt.close()
print('Saved figures inside each question results/plots folder.')
