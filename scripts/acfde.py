import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acfde.pipeline import acfde_pipeline


def default_hparams(dataset: str):
    if dataset == "fiqa":
        return 0.8, 20, 6
    if dataset == "scidocs":
        return 0.1, 300, 5
    return 0.1, 160, 5


def parse_args():
    parser = argparse.ArgumentParser("ACFDE pipeline runner")

    parser.add_argument("--dataset", type=str, default="msmarco", help="Dataset name under dataset root")
    parser.add_argument("--mv-encoder", type=str, default="colbert", help="Multivector encoder folder name")
    parser.add_argument("--sv-encoder", type=str, default="qwen06b", help="Single-vector encoder folder name")

    parser.add_argument("--dataset-root", type=str, default=None, help="Override dataset root")
    parser.add_argument("--output-root", type=str, default=None, help="Override output root")

    parser.add_argument("--sampling-rate", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--mv-pred", type=int, default=200)
    parser.add_argument("--efs", type=int, default=2000)
    parser.add_argument("--proj-d", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--choose-rate", type=float, default=None)
    parser.add_argument("--nrep", type=int, default=None)
    parser.add_argument("--ksim", type=int, default=None)

    parser.add_argument("--hybrid", action="store_true", default=True)
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--hnsw", action="store_true", default=True)
    parser.add_argument("--no-hnsw", action="store_true")

    parser.add_argument("--train-fusion", action="store_true", default=False)
    parser.add_argument("--no-train-fusion", action="store_true")
    parser.add_argument("--fusion-epochs", type=int, default=20)
    parser.add_argument("--fusion-batch-size", type=int, default=256)
    parser.add_argument("--fusion-lr", type=float, default=0.1)
    parser.add_argument("--fusion-num-neg", type=int, default=64)

    return parser.parse_args()


def main():
    args = parse_args()

    choose_rate, nrep, ksim = default_hparams(args.dataset)
    if args.choose_rate is not None:
        choose_rate = args.choose_rate
    if args.nrep is not None:
        nrep = args.nrep
    if args.ksim is not None:
        ksim = args.ksim

    use_hybrid = False if args.no_hybrid else args.hybrid
    use_hnsw = False if args.no_hnsw else args.hnsw
    train_fusion = False if args.no_train_fusion else args.train_fusion

    print("\n===== ACFDE (Public Minimal Project) =====")

    recall = acfde_pipeline(
        dataset=args.dataset,
        sv_encoder=args.sv_encoder,
        mv_encoder=args.mv_encoder,
        hnsw=use_hnsw,
        hybrid=use_hybrid,
        rerank=False,
        mv_pred=args.mv_pred,
        choose_rate=choose_rate,
        sampling_rate=args.sampling_rate,
        efs=args.efs,
        nrep=nrep,
        ksim=ksim,
        proj_d=args.proj_d,
        topk=args.topk,
        train_fusion=train_fusion,
        fusion_epochs=args.fusion_epochs,
        fusion_batch_size=args.fusion_batch_size,
        fusion_lr=args.fusion_lr,
        fusion_num_neg=args.fusion_num_neg,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        seed=args.seed,
    )

    print("\nFinal recall:", recall)


if __name__ == "__main__":
    main()
