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

    def train_validation_test_split(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        random_seed: Optional[int] = None,
        out_path: Optional[Path] = None
    ) -> Tuple[List[Sample], List[Sample], List[Sample]]:
        
        # check ratios
        if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and 0 < test_ratio < 1):
            raise ValueError("Ratios must be between 0 and 1.")
        
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError("Sum of ratios must equal 1")
        
        # random shuffle with seed
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()  
        
        train_split: List[Sample] = []
        val_split: List[Sample] = []
        test_split: List[Sample] = []

        sample_by_label: Dict[int, List[Sample]] = defaultdict(list)
        for record in self.records:
            sample_by_label[record.label].append(record)

        for label, items in sample_by_label.items():
                items = items[:]  # make a copy
                rng.shuffle(items)

                n = len(items)

                raw = [n * train_ratio, n * val_ratio, n * test_ratio]
                floors = [int(x) for x in raw]
                remaining = n - sum(floors)
            
                remainders = sorted(((raw[i] - floors[i], i) for i in range(3)), key=lambda t: (-t[0], t[1]))
                for _, idx in remainders[:remaining]:
                    floors[idx] += 1
                n_train, n_val, n_test = floors

                train_split.extend(items[:n_train])
                val_split.extend(items[n_train:n_train + n_val])
                test_split.extend(items[n_train + n_val: n_train + n_val + n_test])

        val_surplus = len(val_split) - 9000
        if val_surplus > 0:
            # move surplus from val to test
            moved = val_split[:val_surplus]
            test_split.extend(moved)
            val_split = val_split[val_surplus:]
        elif val_surplus < 0:
            # move deficit from test to val
            deficit = -val_surplus
            moved = test_split[:deficit]
            val_split.extend(moved)
            test_split = test_split[deficit:]


        if out_path is not None:
            out_path.mkdir(parents=True, exist_ok=True)
            train_file = out_path / "train_list.txt"
            val_file   = out_path / "val_list.txt"
            test_file  = out_path / "test_list.txt"

            for split, file in zip([train_split, val_split, test_split], [train_file, val_file, test_file]):
                with open(file, 'w') as f:
                    for sample in split:
                        f.write(f"{sample.rel_path} {sample.label}\n")
        
        print(f"Train set size: {len(train_split)}\nValidation set size: {len(val_split)}\nTest set size: {len(test_split)}")

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
