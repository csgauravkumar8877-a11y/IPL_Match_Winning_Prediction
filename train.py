"""Train the IPL chase win-probability ANN and save everything the app needs.

Usage:
    python train.py                       # download data from Cricsheet, train, save to ./artifacts
    python train.py --skip-download       # reuse ./data/ipl_json that is already on disk
    python train.py --epochs 10           # quicker experimental run
"""
import argparse
from pathlib import Path

import ipl_pipeline as ip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ip.DEFAULT_DATA_DIR)
    p.add_argument("--artifact-dir", type=Path, default=ip.DEFAULT_ARTIFACT_DIR)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--skip-download", action="store_true",
                   help="Use JSON files already present in <data-dir>/ipl_json")
    p.add_argument("--force-download", action="store_true", help="Re-download the Cricsheet archive")
    args = p.parse_args()

    report = ip.train_and_save(
        data_dir=args.data_dir, artifact_dir=args.artifact_dir,
        epochs=args.epochs, batch_size=args.batch_size,
        download=not args.skip_download, force_download=args.force_download,
        progress=print,
    )

    m = report["metrics"]
    print("\nTest-set results (held-out matches)")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "brier_score", "log_loss"):
        if m.get(k) is not None:
            print(f"  {k:<12}: {m[k]:.4f}")
    print(f"\nArtifacts written to: {args.artifact_dir}\nNow run:  streamlit run app.py")


if __name__ == "__main__":
    main()
