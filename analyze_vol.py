import re
import numpy as np

filepath = 'xdt/1.txt'
with open(filepath, 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()

data_lines = lines[2:]
all_vals = []
for line in data_lines:
    stripped = line.strip()
    if stripped == '' or stripped.startswith('---') or stripped.startswith('==='):
        continue
    vals = [int(v) for v in stripped.split('\t') if v.strip()]
    all_vals.extend(vals)

arr = np.array(all_vals, dtype=np.int32)
print("1.txt total values:", len(arr))
print("range:", arr.min(), "-", arr.max())
for t in [0, 1, 3, 5, 8, 10, 15, 20, 30, 50]:
    c = int((arr > t).sum())
    p = c / len(arr) * 100
    print(f"  > {t}: {c} ({p:.2f}%)")
