from pathlib import Path

def fix_annotation(label_path: Path, new_label_path: Path):
    label = []
    with label_path.open("r") as f:
        for line in f.readlines():
            line = line.split()
            line = [int(line[0])] + list(map(float, line[1:]))
            label.append(line)
    new_label_path.parent.mkdir(parents=True, exist_ok=True)
    with new_label_path.open("w") as f:
        for l in label:
            line = f"{l[0]} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n"
            f.write(line)

root = Path(r"D:\Nanovare\data\mast_test_dataset_V3\old_labels\test")

for label_path in root.rglob("*.txt"):
    new_label_path = label_path.parents[2] / "labels" / label_path.parent.stem / label_path.name
    fix_annotation(label_path, new_label_path)