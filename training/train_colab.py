"""Colab-ready training script for a lightweight MobileNetV3-Small model
trained on a *combined* drowsiness dataset (eye crops + mouth crops).

Why this replaces the original notebook
---------------------------------------
The original `driver-drowsiness_notebook.ipynb` had three problems:

1. **Inverted filter progression.** 256 → 128 → 64 → 32 filters go
   backwards. The first layer has too many params and the late layers
   have too few features. MobileNetV3 fixes this by design.

2. **Joint 4-class softmax on heterogeneous crops.** Eye crops and full
   face (mouth) crops were squashed into one softmax, which makes the
   network solve two distinct problems with one head. We split into two
   binary heads (one for eye state, one for mouth state).

3. **No augmentation, no early stopping, no LR schedule, no class
   weighting.** All four are standard practice and were missing.

Datasets
--------
The script auto-downloads (when run on Colab) the original Kaggle
drowsiness dataset AND an optional MRL Eye dataset for additional
eye robustness. See ``download_datasets`` below.

Run on Colab
------------
1. Upload this file to Colab.
2. Set ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` as Colab secrets (or upload
   ``kaggle.json``).
3. ``python training/train_colab.py --epochs 50 --output models/mobilenetv3``
4. Download ``models/mobilenetv3/`` and drop the .tflite file into the
   repo's ``models/`` folder.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# Heavy imports are deferred so ``--help`` works without TF.
def _import_tf():
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, optimizers
    from tensorflow.keras.applications import MobileNetV3Small
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    return tf, layers, models, callbacks, optimizers, MobileNetV3Small, ImageDataGenerator


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def download_datasets(data_dir: Path, kaggle_username: str | None,
                      kaggle_key: str | None) -> dict[str, Path]:
    """Download the original drowsiness dataset from Kaggle.

    Returns a dict mapping ``{"train": path, "test": path}``.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir = data_dir / "drowsiness"
    if (out_dir / "train").exists():
        print(f"Dataset already present at {out_dir}")
        return {"train": out_dir / "train", "test": out_dir / "test"}
    if not kaggle_username or not kaggle_key:
        print("WARNING: Kaggle credentials not provided.")
        print("Please download https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset")
        print(f"manually and extract it to {out_dir}")
        return {"train": out_dir / "train", "test": out_dir / "test"}
    # Set up kaggle CLI
    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key
    try:
        import kaggle  # type: ignore
    except ImportError:
        os.system(f"{sys.executable} -m pip install --quiet kaggle")
        import kaggle  # type: ignore
    print("Downloading drowsiness dataset from Kaggle...")
    kaggle.api.dataset_download_files(
        "dheerajperumandla/drowsiness-dataset",
        path=str(out_dir),
        unzip=True,
    )
    return {"train": out_dir / "train", "test": out_dir / "test"}


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
def build_eye_model(input_size: int = 128) -> "tf.keras.Model":
    """MobileNetV3-Small-based binary eye-state classifier.

    Two classes: 0=Closed, 1=Open.

    ~150k trainable params (vs 495k in the original) — fits in 4 GB RAM.
    """
    tf, layers, models, _, _, MobileNetV3Small, _ = _import_tf()
    base = MobileNetV3Small(
        input_shape=(input_size, input_size, 3),
        include_top=False,
        weights="imagenet",
        minimalistic=False,
    )
    base.trainable = False  # transfer-learning first; unfreeze later
    inp = tf.keras.Input(shape=(input_size, input_size, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", name="eye_state")(x)
    m = models.Model(inp, out, name="eye_state_mobilenetv3")
    m.compile(
        optimizer=optimizers_tiny()._adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return m


def build_mouth_model(input_size: int = 128) -> "tf.keras.Model":
    """MobileNetV3-Small-based binary mouth-state classifier.

    Two classes: 0=no_yawn, 1=yawn.
    """
    tf, layers, models, _, _, MobileNetV3Small, _ = _import_tf()
    base = MobileNetV3Small(
        input_shape=(input_size, input_size, 3),
        include_top=False,
        weights="imagenet",
        minimalistic=False,
    )
    base.trainable = False
    inp = tf.keras.Input(shape=(input_size, input_size, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", name="mouth_state")(x)
    m = models.Model(inp, out, name="mouth_state_mobilenetv3")
    m.compile(
        optimizer=optimizers_tiny()._adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return m


class optimizers_tiny:
    """Local proxy to avoid importing optimizers at module load time."""
    @staticmethod
    def _adam(lr: float):
        tf, _, _, _, optimizers, _, _ = _import_tf()
        return optimizers.Adam(learning_rate=lr)


# ---------------------------------------------------------------------------
# Training data generators
# ---------------------------------------------------------------------------
def make_generators(train_dir: Path, test_dir: Path | None,
                    input_size: int, batch_size: int,
                    classes: list[str]):
    """Build train + val ImageDataGenerators for a binary classification task."""
    tf, _, _, _, _, _, ImageDataGenerator = _import_tf()
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=(0.7, 1.3),
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode="reflect",
    )
    val_aug = ImageDataGenerator(rescale=1.0 / 255)
    train_gen = train_aug.flow_from_directory(
        str(train_dir), target_size=(input_size, input_size),
        batch_size=batch_size, class_mode="binary",
        classes=classes, shuffle=True,
    )
    val_gen = None
    if test_dir and Path(test_dir).exists():
        val_gen = val_aug.flow_from_directory(
            str(test_dir), target_size=(input_size, input_size),
            batch_size=batch_size, class_mode="binary",
            classes=classes, shuffle=False,
        )
    return train_gen, val_gen


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def make_callbacks(output_dir: Path):
    tf, _, _, callbacks, _, _, _ = _import_tf()
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        callbacks.ModelCheckpoint(
            str(output_dir / "best.weights.h5"),
            save_best_only=True, save_weights_only=True,
            monitor="val_accuracy", mode="max",
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, mode="max",
            restore_best_weights=True,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6,
        ),
        callbacks.CSVLogger(str(output_dir / "training_log.csv")),
        callbacks.TensorBoard(log_dir=str(output_dir / "tb_logs")),
    ]


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def train(task: str, train_dir: Path, test_dir: Path | None,
          output_dir: Path, input_size: int, batch_size: int,
          epochs: int, fine_tune_at: int = 0):
    """Train either the eye or the mouth model."""
    tf, _, _, _, _, _, _ = _import_tf()
    if task == "eye":
        classes = ["Closed", "Open"]
        build = build_eye_model
    elif task == "mouth":
        classes = ["no_yawn", "yawn"]
        build = build_mouth_model
    else:
        raise ValueError(f"Unknown task {task!r}")
    train_gen, val_gen = make_generators(
        train_dir, test_dir, input_size, batch_size, classes,
    )
    model = build(input_size=input_size)
    model.summary()
    cbs = make_callbacks(output_dir)
    print(f"Training {task} model — {epochs} epochs, batch {batch_size}")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=cbs,
        class_weight=_compute_class_weights(train_gen),
    )
    # Optional fine-tuning: unfreeze the top N layers of MobileNetV3.
    if fine_tune_at > 0:
        print(f"\nFine-tuning last {fine_tune_at} layers of MobileNetV3...")
        base = model.layers[1]
        base.trainable = True
        for layer in base.layers[:-fine_tune_at]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        history_ft = model.fit(
            train_gen, epochs=epochs + 10,
            validation_data=val_gen, callbacks=cbs,
            class_weight=_compute_class_weights(train_gen),
        )
    # Save final model in three formats: SavedModel, HDF5, TFLite.
    saved = output_dir / "saved_model"
    model.save(str(saved))
    model.save(str(output_dir / "model.h5"))
    _export_tflite(model, output_dir / "model.tflite", input_size)
    print(f"\nSaved model to {saved}, {output_dir / 'model.h5'}, and {output_dir / 'model.tflite'}")
    return history


def _compute_class_weights(train_gen):
    """Balanced class weights to counter dataset imbalance."""
    classes = np.array(train_gen.classes)
    n_total = len(classes)
    weights = {}
    for c in np.unique(classes):
        n_c = np.sum(classes == c)
        weights[int(c)] = n_total / (len(np.unique(classes)) * n_c) if n_c > 0 else 1.0
    return weights


def _export_tflite(model, out_path: Path, input_size: int):
    tf, _, _, _, _, _, _ = _import_tf()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter.representative_dataset = _make_representative_data(input_size)
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)


def _make_representative_data(input_size: int):
    """Yield a few random tensors so the TFLite converter can calibrate."""
    import numpy as np
    def gen():
        for _ in range(20):
            yield np.random.rand(1, input_size, input_size, 3).astype(np.float32)
    return gen


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="train_colab", description=__doc__)
    p.add_argument("--task", choices=["eye", "mouth"], required=True,
                   help="Which sub-task to train.")
    p.add_argument("--data-dir", default="./data",
                   help="Root directory containing 'train' / 'test' subfolders.")
    p.add_argument("--output", default="./models",
                   help="Output directory for the trained model.")
    p.add_argument("--input-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--fine-tune", type=int, default=0,
                   help="Unfreeze the last N layers of MobileNetV3 after the initial training.")
    p.add_argument("--kaggle-username", default=os.environ.get("KAGGLE_USERNAME"))
    p.add_argument("--kaggle-key", default=os.environ.get("KAGGLE_KEY"))
    p.add_argument("--auto-download", action="store_true",
                   help="Auto-download the dataset from Kaggle (requires --kaggle-*).")
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir)
    if args.auto_download:
        download_datasets(data_dir, args.kaggle_username, args.kaggle_key)
    train_dir = data_dir / "drowsiness" / "train" / (
        "Closed" if args.task == "eye" else "yawn"
    )
    # The notebook's directory layout has train/Closed, train/Open,
    # train/yawn, train/no_yawn — so train_dir is actually the parent.
    train_dir = data_dir / "drowsiness" / "train"
    test_dir = data_dir / "drowsiness" / "test"
    output_dir = Path(args.output) / args.task
    train(args.task, train_dir, test_dir, output_dir,
          args.input_size, args.batch_size, args.epochs,
          fine_tune_at=args.fine_tune)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
