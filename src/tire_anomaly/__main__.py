import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Single best-model tire anomaly pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create radial maps and raw-coordinate bindings")
    prepare.add_argument("--csv", type=Path, required=True)
    prepare.add_argument("--bmp", type=Path, required=True, help="Organized scan preview, used only for dimensions")
    prepare.add_argument("--output", type=Path, required=True)
    predict = commands.add_parser("infer", help="Run the selected pipeline")
    predict.add_argument("--prepared", type=Path, required=True)
    predict.add_argument("--model-dir", type=Path, default=Path("models"))
    predict.add_argument("--fold", type=int, choices=range(5), required=True)
    predict.add_argument("--fusion-head", choices=("final", "fold"), default="final")
    predict.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    predict.add_argument("--raw-csv", type=Path)
    predict.add_argument("--backbone-source", type=Path)
    predict.add_argument("--backbone-weights", type=Path)
    predict.add_argument("--batch-size", type=int, default=16)
    predict.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        from .prepare import prepare_scan
        prepare_scan(args.csv, args.bmp, args.output)
    else:
        from .pipeline import infer
        infer(args.prepared, args.model_dir, args.fold, args.output, args.device,
              args.raw_csv, args.backbone_source, args.backbone_weights,
              args.fusion_head, args.batch_size)


if __name__ == "__main__":
    main()
