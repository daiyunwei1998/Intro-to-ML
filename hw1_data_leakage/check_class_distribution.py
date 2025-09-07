from pathlib import Path
from typing import Dict
from collections import Counter
import matplotlib.pyplot as plt


def _read_split_file(file_path: Path):
    """Read labels from split file ('rel_path label' format)."""
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                labels.append(int(parts[1]))
    return labels


def plot_split_distributions(split_dir: Path, class_map: Dict[str, int], save_path: Path):
    """
    Plot class distributions for train/val/test splits in one figure with 3 subplots.
    Saves to save_path.
    """
    split_files = {
        "Train": split_dir / "train_list.txt",
        "Validation": split_dir / "val_list.txt",
        "Test": split_dir / "test_list.txt",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    id_to_name = {v: k for k, v in class_map.items()}

    for ax, (split_name, file_path) in zip(axes, split_files.items()):
        if not file_path.exists():
            ax.set_title(f"{split_name} (missing)")
            ax.axis("off")
            continue

        labels = _read_split_file(file_path)
        counter = Counter(labels)
        ordered_ids = sorted(counter.keys())
        x_names = [id_to_name.get(cid, str(cid)) for cid in ordered_ids]
        y = [counter[cid] for cid in ordered_ids]

        bars = ax.bar(x_names, y)
        ax.set_title(split_name)
        ax.set_xlabel("Class")
        if split_name == "Train":
            ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # annotate counts
        total = sum(y)
        for rect, cnt in zip(bars, y):
            pct = 100.0 * cnt / total if total > 0 else 0
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height(),
                f"{cnt}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        if len(x_names) > 8:
            ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved class distribution plot to {save_path}")


def main():
    split_dir = Path("splits")
    save_path = Path("plots/class_distributions.png")
    class_map = {
        "digit_0": 0,
        "digit_1": 1,
        "digit_2": 2,
        "digit_3": 3,
        "digit_4": 4,
        "digit_5": 5,
        "digit_6": 6,
        "digit_7": 7,
        "digit_8": 8,
        "digit_9": 9,
    }

    plot_split_distributions(split_dir, class_map, save_path)


if __name__ == "__main__":
    main()
