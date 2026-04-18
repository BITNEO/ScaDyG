import argparse
import csv
import math
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "dataset_raw" / "mooc"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dataset" / "mooc"


def load_csv_edges(csv_path: Path):
    src = []
    dst = []
    ts = []
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            src.append(int(float(row[1])) - 1)
            dst.append(int(float(row[2])) - 1)
            ts.append(float(row[3]))
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64), np.asarray(ts, dtype=np.float64)


def ensure_dirs(output_dir: Path):
    for name in ["edge_feature", "edge_index", "edge_time", "node_feature"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def save_release_dataset(output_dir: Path, edge_index: np.ndarray, edge_time: np.ndarray, edge_feature: np.ndarray,
                         node_feature: np.ndarray, num_snapshots: int):
    ensure_dirs(output_dir)
    num_edges = edge_time.shape[0]
    chunk_size = math.ceil(num_edges / num_snapshots)
    snapshot_count = 0
    for snapshot_id, start in enumerate(range(0, num_edges, chunk_size)):
        end = min(start + chunk_size, num_edges)
        np.save(output_dir / "edge_index" / f"{snapshot_id}.npy", edge_index[:, start:end])
        np.save(output_dir / "edge_time" / f"{snapshot_id}.npy", edge_time[start:end])
        np.save(output_dir / "edge_feature" / f"{snapshot_id}.npy", edge_feature[start:end])
        np.save(output_dir / "node_feature" / f"{snapshot_id}.npy", node_feature)
        snapshot_count += 1
    return chunk_size, snapshot_count


def compare_with_release(compare_dir: Path, edge_index: np.ndarray, edge_time: np.ndarray, edge_feature: np.ndarray,
                         node_feature: np.ndarray):
    release_edge_index = np.concatenate(
        [np.load(compare_dir / "edge_index" / f"{i}.npy") for i in range(len(list((compare_dir / "edge_index").glob("*.npy"))))],
        axis=1,
    )
    release_edge_time = np.concatenate(
        [np.load(compare_dir / "edge_time" / f"{i}.npy") for i in range(len(list((compare_dir / "edge_time").glob("*.npy"))))]
    )
    release_edge_feature = np.concatenate(
        [np.load(compare_dir / "edge_feature" / f"{i}.npy") for i in range(len(list((compare_dir / "edge_feature").glob("*.npy"))))],
        axis=0,
    )
    release_node_feature = np.load(compare_dir / "node_feature" / "0.npy")
    return {
        "edge_index_match": np.array_equal(edge_index, release_edge_index),
        "edge_time_match": np.array_equal(edge_time, release_edge_time),
        "edge_feature_match": np.array_equal(edge_feature, release_edge_feature),
        "node_feature_match": np.array_equal(node_feature, release_node_feature),
    }


def main():
    parser = argparse.ArgumentParser(description="Reproduce the published ScaDyG mooc dataset exactly.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_RAW_DIR / "ml_mooc.csv", help="Path to ml_mooc.csv")
    parser.add_argument("--edge-npy", type=Path, default=DEFAULT_RAW_DIR / "ml_mooc.npy", help="Path to ml_mooc.npy")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output dataset/mooc directory")
    parser.add_argument("--num-snapshots", type=int, default=100, help="Number of snapshots in the published release")
    parser.add_argument("--compare-dir", type=Path, default=None, help="Optional released dataset/mooc directory to verify against")
    args = parser.parse_args()

    src, dst, edge_time = load_csv_edges(args.csv)
    edge_feature = np.load(args.edge_npy)[1:]

    if edge_feature.shape[0] != edge_time.shape[0]:
        raise ValueError(f"Edge feature rows {edge_feature.shape[0]} != edge rows {edge_time.shape[0]}")

    num_nodes = int(max(src.max(), dst.max()) + 1)
    node_feature = np.ones((num_nodes, 1), dtype=np.float64)
    edge_index = np.stack([src, dst], axis=0)

    chunk_size, snapshot_count = save_release_dataset(
        args.output_dir,
        edge_index=edge_index,
        edge_time=edge_time,
        edge_feature=edge_feature,
        node_feature=node_feature,
        num_snapshots=args.num_snapshots,
    )

    print(f"num_edges={edge_time.shape[0]}")
    print(f"num_nodes={num_nodes}")
    print(f"chunk_size={chunk_size}")
    print(f"snapshot_count={snapshot_count}")

    if args.compare_dir is not None:
        result = compare_with_release(args.compare_dir, edge_index, edge_time, edge_feature, node_feature)
        print(result)


if __name__ == "__main__":
    main()
