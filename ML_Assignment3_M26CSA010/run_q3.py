"""Q3: capacity sweep, clean/denoising autoencoders, reconstruction and feature-classification tests."""

import argparse
import json
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from common import *
from models import ConvAutoencoder, SmallCNN


# ============================================================
# Dataset wrapper for denoising autoencoder
# ============================================================

class Corrupt(Dataset):
    def __init__(self, ds, noise=0.20):
        self.ds = ds
        self.noise = noise

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        noisy_x = (x + torch.randn_like(x) * self.noise).clamp(0, 1)
        return noisy_x, x, y


def corrupt(x):
    """Add Gaussian noise to input images."""
    return (x + torch.randn_like(x) * 0.20).clamp(0, 1)


# ============================================================
# Autoencoder training (reuse saved checkpoint if available)
# ============================================================

def ae_train(model, loader, val_loader, epochs, name, denoise=False, force_train=False):

    model.to(DEVICE)

    model_path = result_dir("Q3", "models") / f"{name}_best.pt"

    # --------------------------------------------------------
    # REUSE EXISTING CHECKPOINT
    # --------------------------------------------------------
    if model_path.exists() and not force_train:
        print(f"\nCheckpoint already exists for {name}")
        print(f"Reusing saved model: {model_path}")

        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        return model

    # --------------------------------------------------------
    # TRAIN FROM SCRATCH
    # --------------------------------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = 1e9
    history = []

    for ep in range(1, epochs + 1):

        # ---------------- TRAIN ----------------
        model.train()
        total = 0
        n = 0

        for batch in loader:

            if denoise:
                x = batch[0].to(DEVICE)
                target = batch[1].to(DEVICE)
            else:
                x = batch[0].to(DEVICE)
                target = batch[0].to(DEVICE)

            opt.zero_grad()

            out = model(x)
            loss = nn.functional.mse_loss(out, target)

            loss.backward()
            opt.step()

            total += loss.item() * len(x)
            n += len(x)

        # ---------------- VALIDATION ----------------
        model.eval()
        val = 0
        vn = 0

        with torch.no_grad():

            for batch in val_loader:

                if denoise:
                    x = batch[0].to(DEVICE)
                    target = batch[1].to(DEVICE)
                else:
                    x = batch[0].to(DEVICE)
                    target = batch[0].to(DEVICE)

                out = model(x)

                val += nn.functional.mse_loss(
                    out, target, reduction="sum"
                ).item()

                vn += target.numel()

        vloss = val / vn

        row = {
            "epoch": ep,
            "train_mse": total / n,
            "val_pixel_mse": vloss
        }

        history.append(row)

        print(
            f"{name} ep {ep}: "
            f"train mse {total/n:.5f} "
            f"val pixel mse {vloss:.5f}"
        )

        # ---------------- SAVE BEST ----------------
        if vloss < best:
            best = vloss
            torch.save(model.state_dict(), model_path)

    # Save training history
    save_json(
        result_dir("Q3", "metrics") / f"{name}_history.json",
        history
    )

    # Load best checkpoint
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )

    model.eval()

    return model


# ============================================================
# Reconstruction metrics
# ============================================================

def recon_metrics(model, loader, noise_input=False):

    model.eval()

    mse_sum = 0
    count = 0
    samples = []

    with torch.no_grad():

        for x, y in loader:

            inp = corrupt(x) if noise_input else x

            out = model(inp.to(DEVICE)).cpu()

            mse_sum += nn.functional.mse_loss(
                out, x, reduction="sum"
            ).item()

            count += x.numel()

            if len(samples) < 8:

                samples.append({
                    "input": inp[:1].cpu().tolist(),
                    "reconstruction": out[:1].tolist(),
                    "target": x[:1].tolist()
                })

    mse = mse_sum / count

    return {
        "pixel_mse": mse,
        "psnr_db": float(
            10 * np.log10(1 / max(mse, 1e-12))
        ),
        "sample_arrays": samples
    }


# ============================================================
# Feature extraction
# ============================================================

def extract_features(model, loader, mode="encoded"):

    model.eval()

    feat = []
    labs = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)

            if mode == "encoded":
                z = model.encode(x)
            else:
                z = model(x)

            feat.append(z.flatten(1).cpu().numpy())
            labs.append(y.numpy())

    return np.concatenate(feat), np.concatenate(labs)


# ============================================================
# Main Q3 workflow
# ============================================================

def main():

    p = argparse.ArgumentParser()

    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--feature-samples", type=int, default=5000)

    # If specified, train AEs again even if checkpoints exist
    p.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain autoencoders instead of reusing saved checkpoints"
    )

    a = p.parse_args()

    seed_all()

    tf = transforms.ToTensor()

    clean_train = datasets.CIFAR10(
        DATA_DIR,
        train=True,
        download=True,
        transform=tf
    )

    clean_test = datasets.CIFAR10(
        DATA_DIR,
        train=False,
        download=True,
        transform=tf
    )

    # --------------------------------------------------------
    # Stratified 45k / 5k train-validation split
    # --------------------------------------------------------

    ys = np.asarray(clean_train.targets)

    ti, vi = train_test_split(
        np.arange(len(ys)),
        test_size=5000,
        random_state=SEED,
        stratify=ys
    )

    tr = DataLoader(
        Subset(clean_train, ti),
        batch_size=a.batch_size,
        shuffle=True,
        num_workers=a.workers
    )

    va = DataLoader(
        Subset(clean_train, vi),
        batch_size=a.batch_size,
        shuffle=False,
        num_workers=a.workers
    )

    te = DataLoader(
        clean_test,
        batch_size=a.batch_size,
        shuffle=False,
        num_workers=a.workers
    )

    # ========================================================
    # 1. CAPACITY SWEEP
    # ========================================================

    capacities = [16, 64, 256]

    cap_models = {}
    sweep = {}

    print("\n========== Q3: AUTOENCODER CAPACITY SWEEP ==========\n")

    for cap in capacities:

        m = ConvAutoencoder(cap)

        cap_models[cap] = ae_train(
            m,
            tr,
            va,
            a.epochs,
            f"q3_clean_ae_b{cap}",
            force_train=a.force_train
        )

        sweep[str(cap)] = recon_metrics(
            cap_models[cap],
            va
        )

    selected = min(
        capacities,
        key=lambda c: sweep[str(c)]["pixel_mse"]
    )

    save_json(
        result_dir("Q3", "metrics") / "q3_capacity_validation.json",
        {
            "validation": sweep,
            "selected_bottleneck": selected
        }
    )

    print(f"\nSelected bottleneck capacity: {selected}")

    clean_ae = cap_models[selected]

    # ========================================================
    # 2. DENOISING AUTOENCODER
    # ========================================================

    print("\n========== Q3: DENOISING AUTOENCODER ==========\n")

    noisy_train = DataLoader(
        Corrupt(Subset(clean_train, ti)),
        batch_size=a.batch_size,
        shuffle=True,
        num_workers=a.workers
    )

    noisy_val = DataLoader(
        Corrupt(Subset(clean_train, vi)),
        batch_size=a.batch_size,
        shuffle=False,
        num_workers=a.workers
    )

    denoise_ae = ae_train(
        ConvAutoencoder(selected),
        noisy_train,
        noisy_val,
        a.epochs,
        "q3_denoising_ae",
        denoise=True,
        force_train=a.force_train
    )

    # ========================================================
    # 3. TEST RECONSTRUCTION EVALUATION
    # ========================================================

    print("\n========== Q3: TEST RECONSTRUCTION ==========\n")

    test_results = {
        "clean_ae_test_by_bottleneck": {
            str(cap): recon_metrics(cap_models[cap], te)
            for cap in capacities
        },

        "selected_clean_ae_clean_test":
            recon_metrics(clean_ae, te),

        "selected_clean_ae_corrupted_input_test":
            recon_metrics(clean_ae, te, noise_input=True),

        "denoising_ae_clean_test":
            recon_metrics(denoise_ae, te),

        "denoising_ae_corrupted_input_test":
            recon_metrics(denoise_ae, te, noise_input=True)
    }

    save_json(
        result_dir("Q3", "metrics") / "q3_reconstruction_test.json",
        test_results
    )

    # ========================================================
    # 4. CLASSIFICATION THROUGH CNN
    # ========================================================

    print("\n========== Q3: CLASSIFICATION EVALUATION ==========\n")

    cls = SmallCNN().to(DEVICE)

    cls_path = (
        result_dir("Q2", "models") /
        "q2_cnn_clean_best.pt"
    )

    if cls_path.exists():

        print("Reusing Q2 clean CNN checkpoint.")

        cls.load_state_dict(
            torch.load(cls_path, map_location=DEVICE)
        )

    else:

        print(
            "Q2 clean CNN checkpoint absent; "
            "training a clean-image classifier for 10 epochs."
        )

        cls, _ = train_classifier(
            cls,
            tr,
            va,
            10,
            name="q3_reuse_clean_cnn_classifier"
        )

        cls.load_state_dict(
            torch.load(
                result_dir("Q3", "models") /
                "q3_reuse_clean_cnn_classifier_best.pt",
                map_location=DEVICE
            )
        )

    cls.eval()

    classification = {}

    def classifier_eval_input(source):

        n = 0
        correct = 0
        conf = []

        with torch.no_grad():

            for x, y in te:

                if source == "clean":
                    inp = x

                elif source == "corrupted":
                    inp = corrupt(x)

                elif source == "clean_ae_reconstruction":
                    inp = clean_ae(x.to(DEVICE)).cpu()

                elif source == "denoising_ae_reconstruction":
                    inp = denoise_ae(
                        corrupt(x).to(DEVICE)
                    ).cpu()

                elif source == "corrupted_reconstructed_by_clean_ae":
                    inp = clean_ae(
                        corrupt(x).to(DEVICE)
                    ).cpu()

                z = cls(inp.to(DEVICE))

                probs = z.softmax(1)

                correct += (
                    z.argmax(1).cpu() == y
                ).sum().item()

                n += len(y)

                conf.extend(
                    probs.max(1).values.cpu().tolist()
                )

        return {
            "accuracy": correct / n,
            "mean_confidence": float(np.mean(conf))
        }

    conditions = [
        "clean",
        "corrupted",
        "clean_ae_reconstruction",
        "denoising_ae_reconstruction",
        "corrupted_reconstructed_by_clean_ae"
    ]

    for key in conditions:

        classification[key] = classifier_eval_input(key)

        print(
            "Classifier condition",
            key,
            classification[key]
        )

    save_json(
        result_dir("Q3", "metrics") /
        "q3_image_classification_test.json",
        classification
    )

    # ========================================================
    # 5. LINEAR PROBES ON LEARNED REPRESENTATIONS
    # ========================================================

    print("\n========== Q3: LINEAR FEATURE PROBES ==========\n")

    n = min(a.feature_samples, len(ti))

    rng = np.random.default_rng(SEED)

    chosen = rng.choice(ti, n, replace=False)

    feature_train_loader = DataLoader(
        Subset(clean_train, chosen),
        batch_size=a.batch_size,
        shuffle=False,
        num_workers=a.workers
    )

    # Clean AE features
    xtr, ytr = extract_features(
        clean_ae,
        feature_train_loader
    )

    xte, yte = extract_features(
        clean_ae,
        te
    )

    # Denoising AE features
    xtr_dn, ytr_dn = extract_features(
        denoise_ae,
        feature_train_loader
    )

    xte_dn, yte_dn = extract_features(
        denoise_ae,
        te
    )

    # --------------------------------------------------------
    # Raw pixel representation
    # --------------------------------------------------------

    raw_tr = []
    raw_y = []
    raw_te = []
    raw_yt = []

    for x, y in feature_train_loader:

        raw_tr.append(x.flatten(1).numpy())
        raw_y.append(y.numpy())

    for x, y in te:

        raw_te.append(x.flatten(1).numpy())
        raw_yt.append(y.numpy())

    raw_tr = np.concatenate(raw_tr)
    raw_y = np.concatenate(raw_y)
    raw_te = np.concatenate(raw_te)
    raw_yt = np.concatenate(raw_yt)

    # --------------------------------------------------------
    # Linear probe function
    # FIXED: Removed unsupported multi_class='auto'
    # --------------------------------------------------------

    def probe(a, b, c, d, label):

        clf = LogisticRegression(
            max_iter=300,
            solver="lbfgs"
        )

        clf.fit(a, b)

        pred = clf.predict(c)

        return {
            "representation": label,
            "test_accuracy": float(
                accuracy_score(d, pred)
            )
        }

    probes = [
        probe(
            raw_tr, raw_y,
            raw_te, raw_yt,
            "raw_pixels"
        ),

        probe(
            xtr, ytr,
            xte, yte,
            "clean_ae_encoded"
        ),

        probe(
            xtr_dn, ytr_dn,
            xte_dn, yte_dn,
            "denoising_ae_encoded"
        )
    ]

    # --------------------------------------------------------
    # Clean representation evaluated on noisy test images
    # --------------------------------------------------------

    clf = LogisticRegression(max_iter=300)

    clf.fit(xtr, ytr)

    xte_noise = []
    y_noise = []

    with torch.no_grad():

        for x, y in te:

            encoded = clean_ae.encode(
                corrupt(x).to(DEVICE)
            )

            xte_noise.append(
                encoded.flatten(1).cpu().numpy()
            )

            y_noise.append(y.numpy())

    probes.append({
        "representation":
            "clean_ae_encoded_with_corrupted_test_input",

        "test_accuracy": float(
            accuracy_score(
                np.concatenate(y_noise),
                clf.predict(np.concatenate(xte_noise))
            )
        )
    })

    # --------------------------------------------------------
    # Save probe results
    # --------------------------------------------------------

    save_json(
        result_dir("Q3", "metrics") /
        "q3_linear_probe_test.json",

        {
            "linear_probe_results": probes,
            "selected_bottleneck": selected,
            "training_feature_subset_size": n
        }
    )

    print("\nQ3 completed successfully!")
    print("Outputs are organized under results/Q3.")


if __name__ == "__main__":
    main()