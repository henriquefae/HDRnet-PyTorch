from pathlib import Path
import sys
from torch.utils.data import DataLoader

FIVEK_REPO = Path(r"E:/henrique/mit-adobe-fivek-dataset")
sys.path.append(str(FIVEK_REPO))

from dataset.fivek import MITAboveFiveK

dataset = MITAboveFiveK(
    root=r"E:/henrique/data",
    split="debug",
    download=False,
    experts=["c"],
)

loader = DataLoader(dataset, batch_size=None, num_workers=0)

item = next(iter(loader))

print("Item keys:")
print(item.keys())

print("\nitem['files']:")
for k, v in item["files"].items():
    print("KEY:", k)
    print("VALUE:", v)
    print("TYPE:", type(v))
    print("-" * 80)