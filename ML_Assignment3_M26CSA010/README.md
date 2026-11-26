# Machine Learning Assignment 3 — M26CSA010

**Roll number:** M26CSA010  
**Seed:** 26010 (all numeric characters of the complete roll number)  
**Dataset:** CIFAR-10; stratified 45,000 train / 5,000 validation split. The official test set is held out for final reporting.

## Folder structure (organized by the assignment questions and deliverables)

```
ML_Assignment3_M26CSA010/
├── data/                         # CIFAR-10 download cache (created by torchvision)
├── results/
│   ├── Q1/
│   │   ├── models/                # MLP checkpoints
│   │   ├── metrics/                # parameter comparison, test metrics, memorization trajectory
│   │   └── plots/                  # training curves and memorization plot
│   ├── Q2/
│   │   ├── models/                 # clean/spatially shuffled CNN and MLP checkpoints
│   │   ├── metrics/                # clean, disturbed, occlusion accuracy/confidence and error outputs
│   │   └── plots/                  # training curves
│   ├── Q3/
│   │   ├── models/                 # autoencoder and reused classifier checkpoints
│   │   ├── metrics/                 # reconstruction, capacity and representation classification metrics
│   │   └── plots/                   # bottleneck validation plot
│   ├── Q4/
│   │   ├── metrics/                 # SVM sweep, test error/boundary samples, clustering statistics
│   │   └── plots/                   # validation accuracy and support-vector fraction sweeps
│   └── report/                      # place final CVPR PDF and source here
├── common.py
├── models.py
├── run_q1_q2.py
├── run_q3.py
├── run_q4.py
├── plot_results.py
├── report_template.tex
├── requirements.txt
└── README.md
```

## Setup
Recommended Python 3.10–3.12. In PowerShell: 
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
CIFAR-10 downloads automatically on first run. CPU users should set `--workers 0`.

## Run order
```powershell
python run_q1_q2.py --epochs 15 --batch-size 128 --workers 0
python run_q3.py --epochs 15 --batch-size 128 --workers 0 --feature-samples 5000
python run_q4.py --subset 5000 --test-samples 10000 --pca-dims 64
python plot_results.py
```

Run Q1/Q2 once, then Q3 (which reuses the clean Q2 CNN if available), then Q4. The SVM sweep may be slow; the assignment allows a fixed stratified training subset, and its size is saved in the results.

## Results to report (assignment deliverables)
- **Q1:** model specification/parameter count, train-validation curves, noisy-label memorization trajectory, selected models' final test accuracies and interpretation.
- **Q2:** model/parameter table; clean, pixel-location shuffled, and occlusion test results; confidence/error analysis; expected-vs-observed discussion.
- **Q3:** bottleneck comparison, chosen clean AE validation error, test reconstruction error for clean and corrupted inputs, denoising training protocol, image reconstructions, clean-vs-corrupt classifier accuracy/confidence, latent-vs-raw linear probes.
- **Q4(a):** at least five C values across three kernels; validation accuracy and support-vector fraction curves; selected kernel/C and final test accuracy.
- **Q4(b):** class pairs, support vectors and nearest-boundary test images; connect examples to test errors.
- **Q4(c):** label-free clustering, cluster-to-class mapping using training only, test contingency table, class-wise SVM-vs-clustering disagreement.

## Important
The code and folder structure do not substitute for experiments. Run the scripts and report only metrics present in saved files. Do not tune using the official test set.

## Report
Start with `report_template.tex` and place the finished report files under `results/report/`. Replace all TODOs with measured results after experiments.
