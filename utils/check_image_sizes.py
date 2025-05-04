"""Quick utility to inspect image resolutions in ./data/{train|test_release}."""
import argparse
from pathlib import Path
import skimage.io as sio
from tqdm import tqdm


def scan_dir(img_root: Path):
    h_min = w_min = 1e9
    h_max = w_max = 0
    for p in tqdm(list(img_root.rglob("*.tif"))):
        try:
            img = sio.imread(p)
        except Exception as e:
            print(f"⚠️  skip {p.name}: {e}")
            continue
        h, w = img.shape[:2]
        h_min, w_min = min(h_min, h), min(w_min, w)
        h_max, w_max = max(h_max, h), max(w_max, w)
    return (h_min, w_min, h_max, w_max)


def main(data_root: Path):
    train_stats = scan_dir(data_root / "train")
    test_stats = scan_dir(data_root / "test_release")

    print("--- Resolution summary (HxW) ---")
    print(
        f"Train  min: {train_stats[0]}x{train_stats[1]}   max: {train_stats[2]}x{train_stats[3]}")
    print(
        f"Test   min: {test_stats[0]}x{test_stats[1]}   max: {test_stats[2]}x{test_stats[3]}")

    # Simple heuristic suggestion for augmentations
    suggested_min = max(256, min(train_stats[0], test_stats[0]) // 64 * 64)
    suggested_max = max(train_stats[2], test_stats[2])
    print("Suggested DataLoader settings →")
    print(
        f"  INPUT.MIN_SIZE_TRAIN: ({suggested_min}, {suggested_min*1.5:.0f}, {suggested_min*1.75:.0f})")
    print(f"  INPUT.MAX_SIZE_TRAIN: {suggested_max}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan dataset image sizes")
    parser.add_argument("--data_root", default="./data", type=str)
    args = parser.parse_args()
    main(Path(args.data_root))
