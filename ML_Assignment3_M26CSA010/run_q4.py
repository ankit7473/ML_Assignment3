"""Q4 Bonus: CPU-friendly SVM sweep and unsupervised clustering on fixed PCA features.

Features:
- Fixed stratified training subset
- Standardization followed by randomized PCA
- Linear, RBF, and polynomial SVM kernels
- C grid: 0.1, 1, 10
- Live heartbeat every 5 seconds during SVM fitting
- Saves validation sweep after every experiment
- Resumes completed sweep entries when configuration matches
- Re-fits selected model and evaluates on test set
"""

import argparse
import json
import time
import threading

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

from common import *


# =============================================================
# LIVE TRAINING HEARTBEAT
# =============================================================

def fit_with_heartbeat(model, X, y, label="SVM"):
    """Display elapsed time while a long model.fit() is running."""

    stop_event = threading.Event()
    start_time = time.time()

    def heartbeat():
        while not stop_event.wait(5):
            elapsed = time.time() - start_time
            print(
                f"  [{label}] Still running... "
                f"{elapsed:.0f} seconds elapsed",
                flush=True
            )

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()

    print(f"  [{label}] Training started...", flush=True)

    try:
        model.fit(X, y)
    finally:
        stop_event.set()
        thread.join()

    elapsed = time.time() - start_time

    print(
        f"  [{label}] FINISHED in {elapsed:.1f} seconds!",
        flush=True
    )

    return model


# =============================================================
# SAVE PROGRESS
# =============================================================

def save_progress(path, config, rows):
    """Save sweep progress after each experiment."""

    save_json(path, {
        "config": config,
        "sweep": rows
    })


# =============================================================
# MAIN
# =============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--subset", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=10000)
    parser.add_argument("--pca-dims", type=int, default=32)
    parser.add_argument("--seed", type=int, default=SEED)

    args = parser.parse_args()

    seed_all(args.seed)

    # =========================================================
    # 1. LOAD CIFAR-10
    # =========================================================

    print("\n========== Q4: LOADING CIFAR-10 ==========\n", flush=True)

    ds = datasets.CIFAR10(
        DATA_DIR,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    te = datasets.CIFAR10(
        DATA_DIR,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    X = np.asarray(ds.data, dtype=np.float32).reshape(
        len(ds), -1
    ) / 255.0

    y = np.asarray(ds.targets)

    Xtest = np.asarray(te.data, dtype=np.float32).reshape(
        len(te), -1
    ) / 255.0

    ytest = np.asarray(te.targets)

    # =========================================================
    # 2. STRATIFIED TRAIN / VALIDATION SPLIT
    # =========================================================

    print("Preparing stratified training/validation split...", flush=True)

    train_idx, val_idx = train_test_split(
        np.arange(len(y)),
        test_size=5000,
        random_state=args.seed,
        stratify=y
    )

    # Fixed stratified training subset
    if args.subset < len(train_idx):

        sub_idx, _ = train_test_split(
            train_idx,
            test_size=len(train_idx) - args.subset,
            random_state=args.seed,
            stratify=y[train_idx]
        )

    else:
        sub_idx = train_idx

    trX = X[sub_idx]
    try_ = y[sub_idx]

    valX = X[val_idx]
    valy = y[val_idx]

    # Limit test samples
    n_test = min(args.test_samples, len(ytest))

    testX_raw = Xtest[:n_test]
    testy = ytest[:n_test]

    print(f"Training subset: {len(trX)} images", flush=True)
    print(f"Validation set: {len(valX)} images", flush=True)
    print(f"Test samples: {n_test}", flush=True)

    # =========================================================
    # 3. STANDARDIZATION
    # =========================================================

    print("\nStandardizing features...", flush=True)

    scaler = StandardScaler()

    trX = scaler.fit_transform(trX)
    valX = scaler.transform(valX)
    testX = scaler.transform(testX_raw)

    # =========================================================
    # 4. PCA
    # =========================================================

    print(
        f"\nRunning randomized PCA ({args.pca_dims} components)...",
        flush=True
    )

    pca = PCA(
        n_components=args.pca_dims,
        random_state=args.seed,
        svd_solver="randomized"
    )

    start = time.time()

    trP = pca.fit_transform(trX)
    valP = pca.transform(valX)
    testP = pca.transform(testX)

    print(
        f"PCA completed in {time.time() - start:.2f} seconds",
        flush=True
    )
    print(f"Training PCA shape: {trP.shape}", flush=True)

    # =========================================================
    # 5. SVM SWEEP
    # =========================================================

    print("\n========== Q4: SVM KERNEL/C SWEEP ==========\n", flush=True)

    grid_C = [0.1, 1, 10]
    kernels = ["linear", "rbf", "poly"]

    sweep_path = (
        result_dir("Q4", "metrics") /
        "q4_svm_sweep_progress.json"
    )

    final_sweep_path = (
        result_dir("Q4", "metrics") /
        "q4_svm_sweep.json"
    )

    config = {
        "subset": len(trP),
        "pca_dims": args.pca_dims,
        "test_samples": n_test,
        "seed": args.seed,
        "kernels": kernels,
        "C_values": grid_C
    }

    # ---------------------------------------------------------
    # Resume saved sweep progress if configuration matches
    # ---------------------------------------------------------

    val_rows = []

    if sweep_path.exists():

        try:
            with open(sweep_path, "r") as f:
                previous = json.load(f)

            if previous.get("config") == config:

                val_rows = previous.get("sweep", [])

                print(
                    f"Resuming sweep: {len(val_rows)} experiments "
                    f"already completed.",
                    flush=True
                )

            else:
                print(
                    "Previous sweep settings differ. Starting a new sweep.",
                    flush=True
                )

        except Exception as e:
            print(f"Could not read previous sweep: {e}", flush=True)

    completed = {
        (row["kernel"], float(row["C"]))
        for row in val_rows
    }

    total_experiments = len(kernels) * len(grid_C)

    experiment_number = len(completed)

    # ---------------------------------------------------------
    # Run SVM experiments
    # ---------------------------------------------------------

    for kernel in kernels:

        for C in grid_C:

            if (kernel, float(C)) in completed:

                print(
                    f"\nSkipping completed experiment: "
                    f"{kernel}, C={C}",
                    flush=True
                )
                continue

            experiment_number += 1

            print(
                f"\n{'='*55}\n"
                f"EXPERIMENT {experiment_number}/{total_experiments}\n"
                f"Kernel: {kernel} | C: {C}\n"
                f"{'='*55}",
                flush=True
            )

            start = time.time()

            clf = SVC(
                C=C,
                kernel=kernel,
                degree=3,
                gamma="scale",
                decision_function_shape="ovr",
                probability=False,
                cache_size=512
            )

            # LIVE HEARTBEAT DURING TRAINING
            fit_with_heartbeat(
                clf,
                trP,
                try_,
                label=f"{kernel} SVM | C={C}"
            )

            # Validation
            print("Predicting validation set...", flush=True)

            vpred = clf.predict(valP)

            vacc = float(
                accuracy_score(valy, vpred)
            )

            svfrac = float(
                len(clf.support_) / len(trP)
            )

            train_acc = float(
                clf.score(trP, try_)
            )

            fit_time = time.time() - start

            row = {
                "kernel": kernel,
                "C": C,
                "validation_accuracy": vacc,
                "support_vector_fraction": svfrac,
                "train_accuracy": train_acc,
                "fit_seconds": fit_time
            }

            val_rows.append(row)

            print("\nEXPERIMENT RESULTS:", flush=True)
            print(row, flush=True)

            # Save after every experiment
            save_progress(
                sweep_path,
                config,
                val_rows
            )

            print("Progress saved successfully.", flush=True)

    # =========================================================
    # 6. SELECT BEST MODEL USING VALIDATION
    # =========================================================

    best_row = max(
        val_rows,
        key=lambda r: r["validation_accuracy"]
    )

    best_kernel = best_row["kernel"]
    best_C = best_row["C"]
    best_val = best_row["validation_accuracy"]

    print("\n========== BEST VALIDATION MODEL ==========\n", flush=True)
    print(best_row, flush=True)

    # ---------------------------------------------------------
    # Refit selected model
    # ---------------------------------------------------------

    print(
        f"\nRefitting best SVM: kernel={best_kernel}, C={best_C}",
        flush=True
    )

    best_model = SVC(
        C=best_C,
        kernel=best_kernel,
        degree=3,
        gamma="scale",
        decision_function_shape="ovr",
        probability=False,
        cache_size=512
    )

    fit_with_heartbeat(
        best_model,
        trP,
        try_,
        label=f"BEST MODEL ({best_kernel}, C={best_C})"
    )

    # =========================================================
    # 7. TEST EVALUATION
    # =========================================================

    print("\nEvaluating test set...", flush=True)

    pred = best_model.predict(testP)

    test_acc = float(
        accuracy_score(testy, pred)
    )

    cm = confusion_matrix(testy, pred)

    print(f"Test accuracy: {test_acc:.4f}", flush=True)

    # =========================================================
    # 8. ERROR / DECISION BOUNDARY ANALYSIS
    # =========================================================

    print("\nPerforming error and boundary analysis...", flush=True)

    scores = best_model.decision_function(testP)

    if scores.ndim == 2:
        margins = np.min(np.abs(scores), axis=1)
    else:
        margins = np.abs(scores)

    errs = np.where(pred != testy)[0]

    near = np.argsort(margins)[:min(100, len(margins))]

    near_errors = [
        int(i)
        for i in near
        if pred[i] != testy[i]
    ][:30]

    sv_indices = best_model.support_[:1000].tolist()

    # Save SVM results
    svm_results = {
        "config": config,
        "sweep": val_rows,
        "selected_kernel": best_kernel,
        "selected_C": best_C,
        "selected_validation_accuracy": best_val,
        "final_test_accuracy": test_acc,
        "test_confusion_matrix": cm.tolist(),
        "test_error_count": int(len(errs)),
        "support_vector_count": int(len(best_model.support_)),
        "support_vector_fraction": float(
            len(best_model.support_) / len(trP)
        ),
        "test_error_indices": errs[:200].tolist(),
        "closest_test_indices": near[:100].tolist(),
        "closest_test_errors": near_errors,
        "support_vector_training_indices": sv_indices,
        "test_samples_used": len(testy),
        "training_subset_size": len(trP)
    }

    save_json(
        final_sweep_path,
        svm_results
    )

    print("\nSVM results saved successfully.", flush=True)

    # =========================================================
    # 9. K-MEANS CLUSTERING
    # =========================================================

    print("\n========== Q4: K-MEANS CLUSTERING ==========\n", flush=True)

    start = time.time()

    km = KMeans(
        n_clusters=10,
        random_state=args.seed,
        n_init=5
    )

    print("KMeans training started...", flush=True)

    tr_cluster = km.fit_predict(trP)

    print("KMeans training finished. Predicting clusters...", flush=True)

    test_cluster = km.predict(testP)

    print(
        f"KMeans completed in {time.time() - start:.2f} seconds",
        flush=True
    )

    # ---------------------------------------------------------
    # Cluster/class contingency table
    # ---------------------------------------------------------

    contingency = np.zeros((10, 10), dtype=int)

    for c, label in zip(tr_cluster, try_):
        contingency[c, label] += 1

    # Majority label mapping
    mapping = contingency.argmax(axis=1)

    cluster_pred = mapping[test_cluster]

    cluster_acc = float(
        accuracy_score(testy, cluster_pred)
    )

    # Hungarian alignment
    r, c = linear_sum_assignment(-contingency)

    hungarian = np.zeros(10, dtype=int)
    hungarian[r] = c

    hung_pred = hungarian[test_cluster]

    hungarian_acc = float(
        accuracy_score(testy, hung_pred)
    )

    # ---------------------------------------------------------
    # SVM vs clustering disagreement
    # ---------------------------------------------------------

    disagreement = (pred != hung_pred)

    disagree_classes = {}

    for cl in range(10):

        mask = testy == cl

        disagree_classes[str(cl)] = {
            "svm_accuracy": float(
                accuracy_score(testy[mask], pred[mask])
            ),
            "cluster_hungarian_accuracy": float(
                accuracy_score(testy[mask], hung_pred[mask])
            ),
            "disagreement_fraction": float(
                disagreement[mask].mean()
            ) if mask.any() else 0.0
        }

    clustering_results = {
        "representation": (
            "standardized raw pixels followed by unsupervised PCA"
        ),
        "pca_dimensions": args.pca_dims,
        "train_contingency_cluster_vs_class": contingency.tolist(),
        "cluster_majority_map": mapping.tolist(),
        "cluster_hungarian_map": hungarian.tolist(),
        "test_cluster_accuracy_majority_mapping": cluster_acc,
        "test_cluster_accuracy_hungarian_mapping": hungarian_acc,
        "svm_test_accuracy_same_test_subset": test_acc,
        "svm_vs_cluster_disagreement_fraction": float(
            disagreement.mean()
        ),
        "per_class": disagree_classes,
        "cluster_test_counts": np.bincount(
            test_cluster,
            minlength=10
        ).tolist()
    }

    save_json(
        result_dir("Q4", "metrics") / "q4_clustering.json",
        clustering_results
    )

    # =========================================================
    # 10. FINAL SUMMARY
    # =========================================================

    print("\n" + "=" * 55, flush=True)
    print("                 Q4 COMPLETED", flush=True)
    print("=" * 55, flush=True)

    print(f"Training samples       : {len(trP)}", flush=True)
    print(f"PCA dimensions        : {args.pca_dims}", flush=True)
    print(f"Test samples          : {len(testy)}", flush=True)
    print(f"SVM kernel           : {best_kernel}", flush=True)
    print(f"SVM C                : {best_C}", flush=True)
    print(f"Best validation acc   : {best_val:.4f}", flush=True)
    print(f"Final test accuracy  : {test_acc:.4f}", flush=True)
    print(f"Support vector count  : {len(best_model.support_)}", flush=True)
    print(
        f"Support vector ratio  : "
        f"{len(best_model.support_) / len(trP):.4f}",
        flush=True
    )
    print(f"KMeans Hungarian acc  : {hungarian_acc:.4f}", flush=True)

    print("=" * 55, flush=True)
    print("\nOutputs saved under results/Q4/metrics", flush=True)


if __name__ == "__main__":
    main()