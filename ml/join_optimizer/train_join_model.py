#!/usr/bin/env python3

import argparse
import csv
import os
import random

import torch
import torch.nn as nn



class JoinOrderModel(nn.Module):

    def __init__(self, input_dim: int = 48, max_tables: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, max_tables),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JoinOrderModelWide(nn.Module):
    def __init__(self, input_dim: int = 48, max_tables: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, max_tables),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JoinOrderModelTableAware(nn.Module):
    def __init__(self, table_feat_dim: int = 3, max_tables: int = 6):
        super().__init__()
        self.table_encoder = nn.Sequential(
            nn.Linear(table_feat_dim, 32), nn.ReLU(), nn.Linear(32, 16),
        )
        
        self.pair_encoder = nn.Sequential(
            nn.Linear(30, 64), nn.ReLU(), nn.Linear(64, 32),
        )
        
        self.head = nn.Sequential(
            nn.Linear(6 * 16 + 32, 64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, max_tables),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        table_feats = x[:, :18].view(-1, 6, 3)
        pair_feats = x[:, 18:]
        
        table_encoded = self.table_encoder(table_feats).view(-1, 6 * 16)
        pair_encoded = self.pair_encoder(pair_feats)
        
        return self.head(torch.cat([table_encoded, pair_encoded], dim=1))


def create_model(variant: str) -> nn.Module:
    if variant == "baseline":
        return JoinOrderModel()
    
    elif variant == "wide":
        return JoinOrderModelWide()
    
    elif variant == "table_aware":
        return JoinOrderModelTableAware()
    
    else:
        raise ValueError(f"Unknown model variant: {variant}")


def list_mle_loss(scores: torch.Tensor, true_order: torch.Tensor, num_tables: torch.Tensor) -> torch.Tensor:
    batch_size = scores.size(0)
    total_loss = torch.tensor(0.0, device = scores.device)

    for b in range(batch_size):
        n = num_tables[b].item()
        s = scores[b]
        pi = true_order[b]

        for i in range(n):
            remaining_idx = pi[i:n]
            remaining = s[remaining_idx]
            max_val = remaining.max().detach()
            total_loss += -(s[pi[i]] - max_val - torch.log(torch.exp(remaining - max_val).sum()))

    return total_loss / batch_size


def exact_match_accuracy(scores: torch.Tensor, true_order: torch.Tensor, num_tables: torch.Tensor) -> float:
    correct = 0
    for b in range(scores.size(0)):
        n = num_tables[b].item()
        pred = torch.argsort(scores[b, :n], descending=True)
        if torch.equal(pred, true_order[b, :n]):
            correct += 1
            
    return correct / scores.size(0)


def load_data(path: str, val_split: float, seed: int):
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        rows = list(reader)

    n = len(rows)
    features = torch.zeros(n, 48)
    num_tables = torch.zeros(n, dtype=torch.long)
    orders = torch.zeros(n, 6, dtype=torch.long)

    for i, row in enumerate(rows):
        for j in range(48):
            features[i, j] = float(row[j])
        num_tables[i] = int(row[48])
        for j in range(6):
            orders[i, j] = int(row[49 + j])

    # Stratified split by num_tables
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for nt in sorted(set(num_tables.tolist())):
        group = [i for i in range(n) if num_tables[i].item() == nt]
        rng.shuffle(group)
        split = int(len(group) * (1 - val_split))
        train_idx.extend(group[:split])
        val_idx.extend(group[split:])

    train_idx = sorted(train_idx)
    val_idx = sorted(val_idx)

    train_features = features[train_idx]
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0)
    std[std == 0] = 1.0 

    features = (features - mean) / std

    return {
        "features": features,
        "num_tables": num_tables,
        "orders": orders,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "mean": mean,
        "std": std,
    }


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data = load_data(args.data, args.val_split, args.seed)
    n_train = len(data["train_idx"])
    n_val = len(data["val_idx"])
    model = create_model(args.model_variant)
    n_params = sum(p.numel() for p in model.parameters())

    print("=== Join Order Model Training ===")
    print(f"Data: {n_train + n_val} samples ({n_train} train, {n_val} val)")
    
    schedule_info = args.lr_schedule
    if args.lr_schedule == "cosine":
        schedule_info += " (eta_min=1e-5)"
        
    print(f"Model: {args.model_variant} ({n_params:,} params)")
    print(f"LR schedule: {schedule_info}, patience: {args.patience}\n")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 1e-5)

    best_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        idx = data["train_idx"][:]
        random.shuffle(idx)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(idx), args.batch_size):
            batch_idx = idx[start : start + args.batch_size]
            feat = data["features"][batch_idx]
            nt = data["num_tables"][batch_idx]
            order = data["orders"][batch_idx]

            scores = model(feat)
            loss = list_mle_loss(scores, order, nt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Evaluate every epoch (cheap with ~600 val samples)
        model.eval()
        with torch.no_grad():
            val_feat = data["features"][data["val_idx"]]
            val_nt = data["num_tables"][data["val_idx"]]
            val_order = data["orders"][data["val_idx"]]

            val_scores = model(val_feat)
            val_loss = list_mle_loss(val_scores, val_order, val_nt).item()
            val_acc = exact_match_accuracy(val_scores, val_order, val_nt)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:>3d}/{args.epochs}  "
                f"train_loss={avg_train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.3f}"
            )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, "join_order_model_best.pt"))
            torch.save({"mean": data["mean"], "std": data["std"], "variant": args.model_variant}, os.path.join(args.output_dir, "feature_stats.pt"))
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step()

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    print(f"\nBest validation accuracy: {best_acc:.3f} (epoch {best_epoch})")
    print(f"Model saved to {os.path.join(args.output_dir, 'join_order_model_best.pt')}")
    print(f"Stats saved to {os.path.join(args.output_dir, 'feature_stats.pt')}")

    if best_acc < 0.8:
        print("\nWARNING: Accuracy below 80% target. Consider more data or tuning hyperparameters.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train join order prediction model")
    parser.add_argument("--data", default="bench/tpch_data/join_training_data.csv", help="Training CSV")
    parser.add_argument("--output-dir", default="ml/join_optimizer/models", help="Output directory")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-variant", choices=["baseline", "wide", "table_aware"], default="baseline", help="Model architecture variant")
    parser.add_argument("--lr-schedule", choices=["none", "cosine"], default="none", help="Learning rate schedule")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience (0 to disable)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
