from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
from collections import defaultdict

class Sample:
    def __init__(self, rel_path: Path, label: int):
        self.rel_path = rel_path  # Relative path
        self.label = label        # Class label


class Dataset:

    def __init__(self, root_dir: Path, class_map: Dict[str, int]):
        self.root_dir = Path(root_dir)
        self.class_map = class_map
        self.records: List[Sample] = [] 
        self.load_data()

    def load_data(self):
        for class_name, label in self.class_map.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            
            for file_path in class_dir.iterdir():
                if file_path.is_file():
                    rel_path = file_path.relative_to(self.root_dir)
                    self.records.append(Sample(rel_path, label))
        
        if not self.records:
            raise ValueError(f"No samples found under {self.root_dir}. ")

    def _hamilton_method(self, total: int, ratios: List[float]) -> List[int]:
        # https://math.libretexts.org/Bookshelves/Applied_Mathematics/Math_in_Society_(Lippman)/04%3A_Apportionment/4.02%3A_Hamiltons_Method
        raw = [total*r for r in ratios]
        quota = [int(r) for r in raw]
        remaining = total - sum(quota)

        order = sorted(
            [(i, raw[i] - quota[i]) for i in range(len(ratios))],
            key=lambda x: (-x[1], x[0])
        )

        for _, idx in order[:remaining]:
            quota[idx] += 1
        return quota
    
    def _split_by_label(self, label_counts: Dict[int, int], ratio: float, target_count:int) -> Dict[int, int]:
        floors = {label: int(count * ratio) for label, count in label_counts.items()}
        base = sum(floors.values())
        need = target_count - base

        if need == 0:
            # perfect match
            return floors
    
        ranked = sorted(
            [(label, count * ratio - floors[label]) for label, count in label_counts.items()],
            key=lambda x: (-x[1], x[0])
        )
        for i in range(need):
            label, _ = ranked[i]
            floors[label] += 1
        return floors

    def train_validation_test_split(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        random_seed: Optional[int] = None,
        out_path: Optional[Path] = None
    ) -> Tuple[List[Sample], List[Sample], List[Sample]]:
        
        # Check ratios
        if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and 0 < test_ratio < 1):
            raise ValueError("Ratios must be between 0 and 1.")
        
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError("Sum of ratios must equal 1")
        
        # set seed
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()  
        
        # Group samples by label
        sample_by_label: Dict[int, List[Sample]] = defaultdict(list)
        for record in self.records:
            sample_by_label[record.label].append(record)

        # Assign group quotas with Hamilton's method
        N = len(self.records)
        T_train, T_val, T_test = self._hamilton_method(N, [train_ratio, val_ratio, test_ratio])

        label_counts: Dict[int, int] = {label: len(items) for label, items in sample_by_label.items()}

        n_train_c = self._split_by_label(label_counts, train_ratio, T_train)
        n_val_c   = self._split_by_label(label_counts, val_ratio, T_val)
        n_test_c  = {c: label_counts[c] - n_train_c[c] - n_val_c[c] for c in label_counts} # take the remainder

        train_split: List[Sample] = []
        val_split:   List[Sample] = []
        test_split:  List[Sample] = []

        for label in sorted(sample_by_label.keys()): # sort by label for reproducibility
            items = sample_by_label[label][:]
            rng.shuffle(items)

            cnt_train = n_train_c[label]
            cnt_val   = n_val_c[label]
            cnt_test  = n_test_c[label]

            train_split.extend(items[:cnt_train])
            val_split.extend(items[cnt_train:cnt_train + cnt_val])
            test_split.extend(items[cnt_train + cnt_val:cnt_train + cnt_val + cnt_test])
        
        # (Optional) Save to files
        if out_path is not None:
            out_path.mkdir(parents=True, exist_ok=True)
            for split, name in [(train_split, "train_list.txt"),
                                (val_split,   "val_list.txt"),
                                (test_split,  "test_list.txt")]:
                with open(out_path / name, "w") as f:
                    for s in split:
                        f.write(f"{s.rel_path} {s.label}\n")

        print(
            f"Train set size: {len(train_split)}\n"
            f"Validation set size: {len(val_split)}\n"
            f"Test set size: {len(test_split)}"
        )
        return train_split, val_split, test_split


    def load_split_from_file(self, file_path: Path) -> List[Sample]:
        samples = []
        with open(file_path, 'r') as f:
            for line in f:
                rel_path_str, label_str = line.strip().split()
                rel_path = Path(rel_path_str)
                label = int(label_str)
                samples.append(Sample(rel_path, label))
        return samples
    
    def check_data_leakage(self, split_dir : Path) -> bool:
        train_samples = self.load_split_from_file(split_dir / "train_list.txt")
        val_samples   = self.load_split_from_file(split_dir / "val_list.txt")
        test_samples  = self.load_split_from_file(split_dir / "test_list.txt")

        train_set = set(s.rel_path for s in train_samples)
        val_set   = set(s.rel_path for s in val_samples)
        test_set  = set(s.rel_path for s in test_samples)

        if train_set.intersection(val_set) or train_set.intersection(test_set) or val_set.intersection(test_set):
            return True

        return False

def main():
    data_dir = Path("MNIST")
    split_dir = Path("splits")

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

    dataset = Dataset(data_dir, class_map)
    dataset.train_validation_test_split(0.7, 0.15, 0.15, random_seed=614001003, out_path=split_dir)
    leakage = dataset.check_data_leakage(split_dir)
    print(f"Data leakage detected: {leakage}")

if __name__ == "__main__":
    main()
