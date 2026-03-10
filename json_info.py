import json
import os

train_data = os.listdir("train_data")
val_data = os.listdir("val_data")

for folder, files in (("train_data", train_data), ("val_data", val_data)):
    print(f"Folder: {folder}")
    for fname in files:
        path = os.path.join(folder, fname)
        if not os.path.isfile(path) or not fname.lower().endswith(".json"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  {fname}: failed to load ({e})")
            continue

        if isinstance(data, list):
            count = len(data)
        elif isinstance(data, dict):
            # if dict values are lists of equal length, treat that as record count
            lists = [v for v in data.values() if isinstance(v, list)]
            if lists and all(len(l) == len(lists[0]) for l in lists):
                count = len(lists[0])
            else:
                count = len(data)
        else:
            count = 1

        print(f"  {fname}: {count}")