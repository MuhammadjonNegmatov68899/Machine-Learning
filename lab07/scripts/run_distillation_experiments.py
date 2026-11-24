"""
Task 4: Knowledge Distillation Experiments.
Trains and compares two identical Student CNN architectures (~100k parameters):
1. Plain Student trained with standard Cross-Entropy loss only.
2. Distilled Student trained with Hinton Distillation Loss (Teacher = 1M Baseline, T=4.0, alpha=0.5).
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import yaml
import pandas as pd
import torch

from lab07.src.utils import set_seed, get_device
from lab07.src.dataset import get_fashion_mnist_dataloaders
from lab07.src.models import build_baseline_model, build_student_model
from lab07.src.training import train_model, train_distilled_student, evaluate
from lab07.src.benchmark import profile_full_model_metrics
from lab07.src.visualization import plot_distillation_comparison


def main():
    config_path = PROJECT_ROOT / "lab07" / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["experiment"].get("seed", 42)
    set_seed(seed)
    device = get_device(config["experiment"].get("device", "auto"))
    print(f"=== Task 4: Knowledge Distillation on {device} ===")

    ckpt_dir = PROJECT_ROOT / "lab07" / "checkpoints"
    results_dir = PROJECT_ROOT / "lab07" / "results"
    plots_dir = PROJECT_ROOT / "lab07" / "plots"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    baseline_ckpt_path = str(ckpt_dir / "baseline_teacher.pt")
    if not os.path.exists(baseline_ckpt_path):
        raise FileNotFoundError(
            f"Teacher checkpoint not found at {baseline_ckpt_path}. Run train_baseline.py first!"
        )

    # 1. Load frozen teacher
    teacher = build_baseline_model(config)
    teacher_ckpt = torch.load(baseline_ckpt_path, map_location="cpu")
    teacher.load_state_dict(teacher_ckpt["model_state_dict"])
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Strictly verify teacher is frozen and functional for inference
    assert all(not p.requires_grad for p in teacher.parameters()), "Teacher parameters must be frozen!"
    with torch.no_grad():
        dummy_test = torch.randn(2, 1, 28, 28, device=device)
        dummy_out = teacher(dummy_test)
        assert dummy_out.shape == (2, 10), f"Teacher output shape {dummy_out.shape} unexpected!"
    print(f"Verified frozen teacher ({teacher.count_parameters():,} parameters, all requires_grad=False).")

    # Data loaders
    data_cfg = config["data"]
    num_workers = 0 if sys.platform == "darwin" else data_cfg.get("num_workers", 2)
    train_loader, val_loader, test_loader = get_fashion_mnist_dataloaders(
        data_dir=data_cfg.get("preferred_data_dir", "data"),
        batch_size=data_cfg.get("batch_size", 128),
        test_batch_size=data_cfg.get("test_batch_size", 256),
        val_split=data_cfg.get("val_split", 0.1),
        seed=seed,
        num_workers=num_workers,
    )

    student_cfg = config["student_model"]
    distill_cfg = config["distillation"]
    epochs = student_cfg.get("epochs", 8)
    lr = float(student_cfg.get("lr", 0.001))
    weight_decay = float(student_cfg.get("weight_decay", 1e-4))
    temperature = float(distill_cfg.get("temperature", 4.0))
    alpha = float(distill_cfg.get("alpha", 0.5))

    # ==========================================
    # 2. Train Plain Student (Cross-Entropy Only)
    # ==========================================
    print("\n--- 1. Training Plain Student (CE Only) ---")
    set_seed(seed)
    plain_student = build_student_model(config).to(device)
    print(f"Initialized Plain StudentCNN with {plain_student.count_parameters():,} parameters.")

    plain_ckpt_path = str(ckpt_dir / "student_plain.pt")
    plain_history_path = results_dir / "student_plain_history.csv"

    if os.path.exists(plain_ckpt_path) and plain_history_path.exists():
        print(f"Found existing Plain Student checkpoint at {plain_ckpt_path}. Verifying validity...")
        try:
            best_plain_ckpt = torch.load(plain_ckpt_path, map_location=device)
            state_key = "model_state_dict" if "model_state_dict" in best_plain_ckpt else "student_state_dict"
            plain_student.load_state_dict(best_plain_ckpt[state_key])
            plain_student.to(device)
            eval_crit = torch.nn.CrossEntropyLoss()
            test_loss, test_acc = evaluate(plain_student, test_loader, eval_crit, device)
            assert test_acc > 0.50, f"Checkpoint evaluation test accuracy {test_acc:.4f} is too low to be valid."
            plain_history_df = pd.read_csv(plain_history_path)
            plain_results = {
                "test_acc": round(test_acc, 4),
                "test_loss": round(test_loss, 4),
                "best_val_acc": round(best_plain_ckpt.get("val_acc", test_acc), 4),
                "total_training_time_sec": float(plain_history_df["epoch_time_sec"].sum()) if "epoch_time_sec" in plain_history_df.columns else 0.0,
                "history": plain_history_df.to_dict("records"),
            }
            print(f"Verified and reused Plain Student: test_acc={plain_results['test_acc']*100:.2f}%, test_loss={plain_results['test_loss']:.4f}")
        except Exception as e:
            print(f"Plain student checkpoint verification failed ({e}). Re-training from scratch...")
            plain_results = train_model(
                model=plain_student,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                device=device,
                save_path=plain_ckpt_path,
            )
            print(
                f"Plain Student training finished in {plain_results['total_training_time_sec']}s. "
                f"Test Accuracy: {plain_results['test_acc']*100:.2f}%."
            )
            plain_history_df = pd.DataFrame(plain_results["history"])
            plain_history_df.to_csv(plain_history_path, index=False)
            best_plain_ckpt = torch.load(plain_ckpt_path, map_location=device)
            plain_student.load_state_dict(best_plain_ckpt["model_state_dict"])
            plain_student.to(device)
    else:
        plain_results = train_model(
            model=plain_student,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            save_path=plain_ckpt_path,
        )
        print(
            f"Plain Student training finished in {plain_results['total_training_time_sec']}s. "
            f"Test Accuracy: {plain_results['test_acc']*100:.2f}%."
        )
        plain_history_df = pd.DataFrame(plain_results["history"])
        plain_history_df.to_csv(plain_history_path, index=False)
        best_plain_ckpt = torch.load(plain_ckpt_path, map_location=device)
        plain_student.load_state_dict(best_plain_ckpt["model_state_dict"])
        plain_student.to(device)

    plain_metrics = profile_full_model_metrics(
        model=plain_student,
        name="StudentCNN_Plain_CE",
        device=device,
        cpu_latency=True,
    )
    plain_metrics["test_acc"] = plain_results["test_acc"]
    plain_metrics["test_loss"] = plain_results["test_loss"]
    plain_metrics["best_val_acc"] = plain_results["best_val_acc"]
    plain_metrics["total_training_time_sec"] = plain_results["total_training_time_sec"]

    # Release memory before training distilled student
    import gc
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    # ==========================================
    # 3. Train Distilled Student (Hinton KD)
    # ==========================================
    print("\n--- 2. Training Distilled Student (Knowledge Distillation) ---")
    set_seed(seed)
    distill_student = build_student_model(config).to(device)
    print(f"Initialized Distilled StudentCNN with {distill_student.count_parameters():,} parameters.")

    distill_ckpt_path = str(ckpt_dir / "student_distilled.pt")
    distill_history_path = results_dir / "student_distilled_history.csv"

    if os.path.exists(distill_ckpt_path) and distill_history_path.exists():
        print(f"Found existing Distilled Student checkpoint at {distill_ckpt_path}. Verifying validity...")
        try:
            best_distill_ckpt = torch.load(distill_ckpt_path, map_location=device)
            state_key = "student_state_dict" if "student_state_dict" in best_distill_ckpt else "model_state_dict"
            distill_student.load_state_dict(best_distill_ckpt[state_key])
            distill_student.to(device)
            eval_crit = torch.nn.CrossEntropyLoss()
            test_loss, test_acc = evaluate(distill_student, test_loader, eval_crit, device)
            assert test_acc > 0.50, f"Distilled checkpoint evaluation test accuracy {test_acc:.4f} is too low to be valid."
            distill_history_df = pd.read_csv(distill_history_path)
            distill_results = {
                "test_acc": round(test_acc, 4),
                "test_loss": round(test_loss, 4),
                "best_val_acc": round(best_distill_ckpt.get("val_acc", test_acc), 4),
                "total_training_time_sec": float(distill_history_df["epoch_time_sec"].sum()) if "epoch_time_sec" in distill_history_df.columns else 0.0,
                "history": distill_history_df.to_dict("records"),
            }
            print(f"Verified and reused Distilled Student: test_acc={distill_results['test_acc']*100:.2f}%, test_loss={distill_results['test_loss']:.4f}")
        except Exception as e:
            print(f"Distilled student checkpoint verification failed ({e}). Re-training from scratch...")
            distill_results = train_distilled_student(
                student=distill_student,
                teacher=teacher,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                temperature=temperature,
                alpha=alpha,
                device=device,
                save_path=distill_ckpt_path,
            )
            print(
                f"Distilled Student training finished in {distill_results['total_training_time_sec']}s. "
                f"Test Accuracy: {distill_results['test_acc']*100:.2f}%."
            )
            distill_history_df = pd.DataFrame(distill_results["history"])
            distill_history_df.to_csv(distill_history_path, index=False)
            best_distill_ckpt = torch.load(distill_ckpt_path, map_location=device)
            distill_student.load_state_dict(best_distill_ckpt["student_state_dict"])
            distill_student.to(device)
    else:
        distill_results = train_distilled_student(
            student=distill_student,
            teacher=teacher,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            temperature=temperature,
            alpha=alpha,
            device=device,
            save_path=distill_ckpt_path,
        )
        print(
            f"Distilled Student training finished in {distill_results['total_training_time_sec']}s. "
            f"Test Accuracy: {distill_results['test_acc']*100:.2f}%."
        )
        distill_history_df = pd.DataFrame(distill_results["history"])
        distill_history_df.to_csv(distill_history_path, index=False)
        best_distill_ckpt = torch.load(distill_ckpt_path, map_location=device)
        distill_student.load_state_dict(best_distill_ckpt["student_state_dict"])
        distill_student.to(device)
    distill_metrics = profile_full_model_metrics(
        model=distill_student,
        name="StudentCNN_Distilled_KD",
        device=device,
        cpu_latency=True,
    )
    distill_metrics["test_acc"] = distill_results["test_acc"]
    distill_metrics["test_loss"] = distill_results["test_loss"]
    distill_metrics["best_val_acc"] = distill_results["best_val_acc"]
    distill_metrics["total_training_time_sec"] = distill_results["total_training_time_sec"]
    distill_metrics["temperature"] = temperature
    distill_metrics["alpha"] = alpha

    # Save summary comparison
    comparison = {
        "plain_student": plain_metrics,
        "distilled_student": distill_metrics,
        "accuracy_delta_percent": round(
            (distill_metrics["test_acc"] - plain_metrics["test_acc"]) * 100.0, 3
        ),
    }

    comp_path = results_dir / "distillation_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nDistillation Comparison Summary:")
    print(f"  Plain Student Test Acc:     {plain_metrics['test_acc']*100:.2f}%")
    print(f"  Distilled Student Test Acc: {distill_metrics['test_acc']*100:.2f}%")
    print(f"  Accuracy Improvement:       {comparison['accuracy_delta_percent']:+.2f}%")

    # Plot distillation comparison
    print("Generating distillation learning curves plot...")
    plot_distillation_comparison(
        plain_results["history"],
        distill_results["history"],
        str(plots_dir / "knowledge_distillation_comparison.png"),
    )

    print("Task 4 experiments completed successfully!")


if __name__ == "__main__":
    main()
