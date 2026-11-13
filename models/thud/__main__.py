"""Local acoustic-feature and task-trained deep-learning research commands."""

import argparse
import json
import sys

from .audio import FEATURE_NAMES, analyze_wav
from .models import evaluate, fit_model, load_dataset, predict_model, validate_model


def _deep_commands(commands):
    train = commands.add_parser("deep-train", help="Train the task-specific log-mel CNN")
    train.add_argument("csv")
    train.add_argument("--task", choices=("tile", "fill"), required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--seed", type=int, default=0)

    evaluation = commands.add_parser("deep-evaluate", help="Evaluate the CNN on held-out groups")
    evaluation.add_argument("csv")
    evaluation.add_argument("--task", choices=("tile", "fill"), required=True)
    evaluation.add_argument("--group-by", choices=("object_id", "session_id"), required=True)
    evaluation.add_argument("--epochs", type=int, default=40)
    evaluation.add_argument("--seed", type=int, default=0)

    prediction = commands.add_parser("deep-predict", help="Predict with a trained CNN checkpoint")
    prediction.add_argument("model")
    prediction.add_argument("wav")
    prediction.add_argument("--family")

    export = commands.add_parser("deep-export", help="Export a CNN checkpoint to TorchScript")
    export.add_argument("model")
    export.add_argument("--output", required=True)

    yamnet_train = commands.add_parser("yamnet-train", help="Train a head on official YAMNet embeddings")
    yamnet_train.add_argument("checkpoint")
    yamnet_train.add_argument("csv")
    yamnet_train.add_argument("--task", choices=("tile", "fill"), required=True)
    yamnet_train.add_argument("--output", required=True)

    yamnet_predict = commands.add_parser("yamnet-predict", help="Predict with an official YAMNet transfer head")
    yamnet_predict.add_argument("checkpoint")
    yamnet_predict.add_argument("model")
    yamnet_predict.add_argument("wav")
    yamnet_predict.add_argument("--family")

    yamnet_evaluate = commands.add_parser("yamnet-evaluate", help="Evaluate frozen official embeddings on held-out groups")
    yamnet_evaluate.add_argument("checkpoint")
    yamnet_evaluate.add_argument("csv")
    yamnet_evaluate.add_argument("--task", choices=("tile", "fill"), required=True)
    yamnet_evaluate.add_argument("--group-by", choices=("object_id", "session_id"), required=True)

    yamnet_export = commands.add_parser("yamnet-export", help="Export official YAMNet and optional task head to builtin-only TFLite")
    yamnet_export.add_argument("checkpoint")
    yamnet_export.add_argument("--model", help="Optional trained YAMNet head JSON")
    yamnet_export.add_argument("--output", required=True)

    fetch = commands.add_parser("efficientat-fetch", help="Fetch pinned upstream EfficientAT inference sources")
    fetch.add_argument("source_dir")
    for name in ("efficientat-train", "efficientat-evaluate", "efficientat-predict"):
        command = commands.add_parser(name, help="Use official frozen EfficientAT embeddings")
        command.add_argument("checkpoint")
        command.add_argument("--source-dir", required=True)
        if name == "efficientat-predict":
            command.add_argument("model")
            command.add_argument("wav")
            command.add_argument("--family")
        else:
            command.add_argument("csv")
            command.add_argument("--task", choices=("tile", "fill"), required=True)
            if name == "efficientat-train":
                command.add_argument("--output", required=True)
            else:
                command.add_argument("--group-by", choices=("object_id", "session_id"), required=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="thud", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="Extract features from one isolated tap WAV")
    analyze.add_argument("wav")

    eval_cmd = commands.add_parser("evaluate", help="Evaluate whole-object or session holdouts")
    eval_cmd.add_argument("csv")
    eval_cmd.add_argument("--task", choices=("tile", "fill"), required=True)
    eval_cmd.add_argument("--group-by", choices=("object_id", "session_id"), required=True)

    training = commands.add_parser("train", help="Train a portable JSON linear model")
    training.add_argument("csv")
    training.add_argument("--task", choices=("tile", "fill"), required=True)
    training.add_argument("--output", required=True)

    prediction = commands.add_parser("predict", help="Predict using a validated JSON model")
    prediction.add_argument("model")
    prediction.add_argument("wav")
    prediction.add_argument("--family")
    _deep_commands(commands)

    args = parser.parse_args(argv)

    try:
        if args.command == "analyze":
            result = analyze_wav(args.wav)
        elif args.command == "evaluate":
            records, X, y = load_dataset(args.csv, args.task)
            result = evaluate(records, X, y, args.task, args.group_by)
        elif args.command == "train":
            records, X, y = load_dataset(args.csv, args.task)
            family = records[0]["family"] if args.task == "fill" else None
            model = fit_model(X, y, args.task, family=family)
            serialized = json.dumps(model, allow_nan=False, indent=2)
            with open(args.output, "x", encoding="utf-8") as output:
                output.write(serialized + "\n")
            result = {"task": args.task, "output": args.output, "samples": len(records)}
        elif args.command == "predict":
            with open(args.model, encoding="utf-8") as source:
                model = json.load(source)
            validate_model(model)
            if model["task"] == "fill":
                if args.family != model["family"]:
                    raise ValueError("fill prediction requires --family matching the model family")
            elif args.family is not None:
                raise ValueError("tile prediction does not accept --family")
            analysis = analyze_wav(args.wav)
            X = [[analysis["features"][name] for name in FEATURE_NAMES]]
            value = predict_model(model, X).tolist()[0]
            result = {"task": model["task"], "prediction": value}
            if model["task"] == "fill":
                result["family"] = model["family"]
        elif args.command == "deep-train":
            from .deep import load_deep_dataset, save_checkpoint, train_deep
            records, waveforms, targets, family = load_deep_dataset(args.csv, args.task)
            checkpoint = train_deep(
                waveforms, targets, args.task, family=family, epochs=args.epochs, seed=args.seed,
            )
            save_checkpoint(checkpoint, args.output)
            result = {"task": args.task, "output": args.output, "samples": len(records),
                      "parameter_count": checkpoint["parameter_count"], "epochs": args.epochs}
        elif args.command == "deep-evaluate":
            from .deep import evaluate_deep, load_deep_dataset
            records, waveforms, targets, _ = load_deep_dataset(args.csv, args.task)
            result = evaluate_deep(
                records, waveforms, targets, args.task, args.group_by,
                epochs=args.epochs, seed=args.seed,
            )
        elif args.command == "deep-predict":
            from .deep import load_checkpoint, load_deep_waveform, predict_deep
            checkpoint = load_checkpoint(args.model)
            if checkpoint["task"] == "fill" and args.family != checkpoint["family"]:
                raise ValueError("fill prediction requires --family matching the model family")
            if checkpoint["task"] == "tile" and args.family is not None:
                raise ValueError("tile prediction does not accept --family")
            value = predict_deep(checkpoint, load_deep_waveform(args.wav))[0]
            prediction = ("debonded" if int(value) else "bonded") if checkpoint["task"] == "tile" else float(value)
            result = {"task": checkpoint["task"], "prediction": prediction}
            if checkpoint["task"] == "fill":
                result["family"] = checkpoint["family"]
        elif args.command == "yamnet-train":
            from .pretrained import build_yamnet_transfer_dataset
            from .transfer import fit_transfer_head, save_transfer_head
            _, records, embeddings, labels = build_yamnet_transfer_dataset(args.checkpoint, args.csv, args.task)
            family = records[0]["family"] if args.task == "fill" else None
            head = fit_transfer_head(embeddings, labels, args.task, family=family)
            save_transfer_head(head, args.output)
            result = {"task": args.task, "output": args.output, "samples": len(records),
                      "embedding_dim": head["embedding_dim"], "encoder": "YAMNet"}
        elif args.command == "yamnet-evaluate":
            from .pretrained import build_yamnet_transfer_dataset
            from .transfer import evaluate_transfer_head
            _, records, embeddings, labels = build_yamnet_transfer_dataset(args.checkpoint, args.csv, args.task)
            result = evaluate_transfer_head(records, embeddings, labels, args.task, args.group_by)
        elif args.command == "yamnet-predict":
            from .pretrained import load_yamnet, yamnet_wav_embedding
            from .transfer import load_transfer_head, predict_transfer_head
            head = load_transfer_head(args.model)
            if head["task"] == "fill" and args.family != head["family"]:
                raise ValueError("fill prediction requires --family matching the model family")
            if head["task"] == "tile" and args.family is not None:
                raise ValueError("tile prediction does not accept --family")
            embedding = yamnet_wav_embedding(load_yamnet(args.checkpoint), args.wav)
            value = predict_transfer_head(head, embedding[None, :], encoder="yamnet")[0]
            result = {"task": head["task"], "prediction": str(value) if head["task"] == "tile" else float(value),
                      "encoder": "YAMNet"}
            if head["task"] == "fill":
                result["family"] = head["family"]
        elif args.command == "yamnet-export":
            from .tflite import export_yamnet_tflite
            from .transfer import load_transfer_head
            head = load_transfer_head(args.model) if args.model else None
            result = export_yamnet_tflite(args.checkpoint, args.output, head=head)
        elif args.command == "efficientat-fetch":
            from .efficientat import fetch_efficientat_source, EFFICIENTAT_SOURCE_COMMIT
            source = fetch_efficientat_source(args.source_dir)
            result = {"source_dir": str(source), "commit": EFFICIENTAT_SOURCE_COMMIT}
        elif args.command in ("efficientat-train", "efficientat-evaluate", "efficientat-predict"):
            import numpy as np
            from .efficientat import load_efficientat, efficientat_wav_embedding
            from .transfer import fit_transfer_head, evaluate_transfer_head, load_transfer_head, save_transfer_head, predict_transfer_head
            encoder = "efficientat-mn04-as"
            if args.command == "efficientat-predict":
                head = load_transfer_head(args.model)
                if head["encoder"] != encoder:
                    raise ValueError("head encoder does not match EfficientAT")
                if head["task"] == "fill" and args.family != head["family"]:
                    raise ValueError("fill prediction requires --family matching the model family")
                if head["task"] == "tile" and args.family is not None:
                    raise ValueError("tile prediction does not accept --family")
                bundle = load_efficientat(args.checkpoint, args.source_dir)
                embedding = efficientat_wav_embedding(bundle, args.wav)
                value = predict_transfer_head(head, embedding[None, :], encoder=encoder)[0]
                result = {"task": head["task"], "encoder": encoder,
                          "prediction": str(value) if head["task"] == "tile" else float(value)}
                if head["task"] == "fill":
                    result["family"] = head["family"]
            else:
                records, _, labels = load_dataset(args.csv, args.task)
                bundle = load_efficientat(args.checkpoint, args.source_dir)
                embeddings = np.stack([efficientat_wav_embedding(bundle, row["path"]) for row in records])
                if args.command == "efficientat-evaluate":
                    result = evaluate_transfer_head(records, embeddings, labels, args.task, args.group_by, encoder=encoder)
                else:
                    head = fit_transfer_head(embeddings, labels, args.task, records[0].get("family"), encoder=encoder)
                    save_transfer_head(head, args.output)
                    result = {"task": args.task, "output": args.output, "samples": len(records), "encoder": encoder,
                              "embedding_dim": head["embedding_dim"]}
        elif args.command == "deep-export":
            from .deep import export_torchscript, load_checkpoint
            checkpoint = load_checkpoint(args.model)
            export_torchscript(checkpoint, args.output)
            result = {"task": checkpoint["task"], "output": args.output, "format": "torchscript"}
        else:
            raise ValueError(f"unknown command: {args.command}")
        print(json.dumps(result, allow_nan=False))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"thud: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
