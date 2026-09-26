"""Train the portfolio-inspired three-class model on a local CSV."""

import argparse
import json
from pathlib import Path

import numpy as np

from .stock import classification_metrics, chronological_split, load_closes, make_examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="CSV with Date and Close columns")
    parser.add_argument("--close-column", default="Close")
    parser.add_argument("--date-column", default="Date")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=Path("outputs/stock"))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("Install the stock extra: python -m pip install -e '.[stock]'") from exc
    tf.keras.utils.set_random_seed(42)
    closes = load_closes(args.csv, args.close_column, args.date_column)
    x, y = make_examples(closes)
    x_train, x_test, y_train, y_test = chronological_split(x, y, args.train_fraction)
    # Center/scale on training data only. No future observations enter preprocessing.
    mean, std = float(np.mean(x_train)), float(np.std(x_train))
    std = std if std > 0 else 1.0
    x_train, x_test = (x_train - mean) / std, (x_test - mean) / std

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=args.epochs, shuffle=False, verbose=0)
    predictions = np.argmax(model.predict(x_test, verbose=0), axis=1)
    metrics = classification_metrics(y_test, predictions)
    metrics.update({"training_examples": len(y_train), "train_fraction": args.train_fraction,
                    "epochs": args.epochs, "feature": "single daily log return",
                    "scaler_train_mean": mean, "scaler_train_std": std,
                    "note": "Single chronological holdout; no backtest or financial return claim."})
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(args.output / "model.keras")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

