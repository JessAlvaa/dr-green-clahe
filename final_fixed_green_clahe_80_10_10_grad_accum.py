

# %%
import os
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_CACHE_DIR = Path.cwd() / ".cache"
PROJECT_TEMP_DIR = PROJECT_CACHE_DIR / "tmp"
for cache_dir in [PROJECT_CACHE_DIR, PROJECT_TEMP_DIR]:
    cache_dir.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TMP", str(PROJECT_TEMP_DIR))
os.environ.setdefault("TEMP", str(PROJECT_TEMP_DIR))
os.environ.setdefault("TMPDIR", str(PROJECT_TEMP_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("KERAS_HOME", str(PROJECT_CACHE_DIR / "keras"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_CACHE_DIR))
os.environ["CUDA_CACHE_PATH"] = "/mnt/d/DR/.cache/nv"

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW

# %%
@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    img_size: int = 300
    batch_size: int = 32
    phase1_batch_size: int = 32
    phase2_batch_size: int = 8
    phase2_gradient_accumulation_steps: int = 4
    num_classes: int = 5
    label_col: str = "level"

    # Validity first: corrected split.
    split_strategy: str = "patient_grouped"  # "patient_grouped" 
    train_size: float = 0.80
    val_size: float = 0.10
    test_size: float = 0.10

    # Fixed green-channel CLAHE: resize + BGR2RGB + CLAHE on green channel + float32.
    preprocessing_strategy: str = "green_clahe_only"
    augmentation_strategy: str = "online_light"  # "none" or "online_light"
    imbalance_strategy: str = "none"  # "none" or "class_weight"
    loss_name: str = "sparse_categorical_crossentropy"

    # Training.
    backbone_name: str = "EfficientNetB3"
    phase1_epochs: int = 5
    phase2_epochs: int = 30
    lr_phase1: float = 1e-4
    lr_phase2: float = 1e-5
    weight_decay_phase1: float = 1e-4
    weight_decay_phase2: float = 1e-5
    freeze_backbone_batchnorm: bool = True

    # Paths relative to BASE_DIR.
    image_dir_name: str = "resized train 15"
    labels_csv_name: str = "trainLabels15.csv"
    output_dir_name: str = "final_fixed_green_clahe_80_10_10_runtime_rerun2_results"


CFG = ExperimentConfig()

# %%
BASE_DIR = Path.cwd()
IMAGE_DIR = BASE_DIR / CFG.image_dir_name
CSV_PATH = BASE_DIR / CFG.labels_csv_name
OUTPUT_DIR = BASE_DIR / CFG.output_dir_name
MANIFEST_DIR = OUTPUT_DIR / "manifests"
PLOT_DIR = OUTPUT_DIR / "plots"
MODEL_DIR = OUTPUT_DIR / "models"

if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        f"Output directory already exists and is not empty: {OUTPUT_DIR}. "
        "Use a new output_dir_name to avoid overwriting an existing experiment."
    )

for directory in [OUTPUT_DIR, MANIFEST_DIR, PLOT_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

# %%
def set_global_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_global_determinism(CFG.seed)

gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise RuntimeError(
        "No GPU detected by TensorFlow. Use the DR GPU kernel/environment before training."
    )

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("TensorFlow GPUs:", gpus)
print("Config:", json.dumps(asdict(CFG), indent=2))

# %% [markdown]
# ## Dataset loading and leakage-safe splitting

# %%
def extract_patient_id(image_name: str) -> str:
    """EyePACS names are typically '<patient_id>_left/right'."""
    stem = Path(str(image_name)).stem
    if stem.endswith("_left"):
        return stem[:-5]
    if stem.endswith("_right"):
        return stem[:-6]
    return stem


def load_eyepacs_csv(csv_path: Path, image_dir: Path, ext: str = ".jpg") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    if "image" not in df.columns or CFG.label_col not in df.columns:
        raise ValueError(f"Expected columns 'image' and '{CFG.label_col}' in {csv_path}")

    df["image"] = df["image"].astype(str)
    df["filepath"] = df["image"].apply(
        lambda x: str(image_dir / (x if x.endswith(ext) else x + ext))
    )
    df["exists"] = df["filepath"].apply(os.path.exists)
    missing = df.loc[~df["exists"], "filepath"].tolist()
    if missing:
        raise FileNotFoundError(f"{len(missing)} image files are missing. First missing: {missing[0]}")

    df["patient_id"] = df["image"].apply(extract_patient_id)
    df[CFG.label_col] = df[CFG.label_col].astype(int)
    return df[["image", "patient_id", "filepath", CFG.label_col]].reset_index(drop=True)


df_all = load_eyepacs_csv(CSV_PATH, IMAGE_DIR)
print(f"Loaded {len(df_all)} images from {CSV_PATH}")
print("Image-level class distribution:")
print(df_all[CFG.label_col].value_counts().sort_index())
print(f"Unique patient IDs: {df_all['patient_id'].nunique()}")
print(
    "Patients with different labels between eyes:",
    int((df_all.groupby("patient_id")[CFG.label_col].nunique() > 1).sum()),
)

# %%
def legacy_image_split(df: pd.DataFrame, cfg: ExperimentConfig):
    """Original baseline split. Kept only for reproducibility and comparison."""
    train_df, temp_df = train_test_split(
        df,
        test_size=cfg.val_size + cfg.test_size,
        random_state=cfg.seed,
        stratify=df[cfg.label_col],
    )
    relative_test_size = cfg.test_size / (cfg.val_size + cfg.test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=cfg.seed,
        stratify=temp_df[cfg.label_col],
    )
    return train_df, val_df, test_df


def patient_grouped_split(df: pd.DataFrame, cfg: ExperimentConfig):
    """Split by patient ID so left/right eyes cannot cross train/val/test boundaries."""
    groups = df["patient_id"].values

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=cfg.val_size + cfg.test_size,
        random_state=cfg.seed,
    )
    train_idx, temp_idx = next(first_split.split(df, groups=groups))
    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    relative_test_size = cfg.test_size / (cfg.val_size + cfg.test_size)
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_test_size,
        random_state=cfg.seed,
    )
    val_idx, test_idx = next(second_split.split(temp_df, groups=temp_df["patient_id"].values))
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    return train_df, val_df, test_df


def make_splits(df: pd.DataFrame, cfg: ExperimentConfig):
    if cfg.split_strategy == "patient_grouped":
        return patient_grouped_split(df, cfg)
    if cfg.split_strategy == "legacy_image":
        return legacy_image_split(df, cfg)
    raise ValueError(f"Unknown split_strategy: {cfg.split_strategy}")


train_df, val_df, test_df = make_splits(df_all, CFG)
train_df = train_df.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# %%
def summarize_split(df: pd.DataFrame, name: str) -> None:
    print(f"\n--- {name} ---")
    print(f"Images: {len(df)} | Patients: {df['patient_id'].nunique()}")
    print(df[CFG.label_col].value_counts().sort_index())


def assert_no_patient_overlap(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    split_map = {"train": train, "val": val, "test": test}
    for left_name, right_name in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = set(split_map[left_name]["patient_id"]) & set(split_map[right_name]["patient_id"])
        if overlap:
            raise AssertionError(
                f"Patient leakage detected between {left_name} and {right_name}: "
                f"{len(overlap)} overlapping patient IDs"
            )


summarize_split(train_df, "TRAIN")
summarize_split(val_df, "VAL")
summarize_split(test_df, "TEST")

if CFG.split_strategy == "patient_grouped":
    assert_no_patient_overlap(train_df, val_df, test_df)
    print("\nPatient-overlap check passed.")
else:
    print("\nWARNING: legacy_image split can leak patient identity across splits.")

# %%
def save_manifest(df: pd.DataFrame, split_name: str) -> None:
    manifest_path = MANIFEST_DIR / f"{split_name}.csv"
    df[["image", "patient_id", "filepath", CFG.label_col]].to_csv(manifest_path, index=False)
    print(f"Saved {split_name} manifest: {manifest_path}")


for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    save_manifest(split_df, split_name)

with open(MANIFEST_DIR / "config.json", "w") as f:
    json.dump(asdict(CFG), f, indent=2)

# %% [markdown]
# ## Preprocessing and augmentation
#
# ### Fixed green-channel CLAHE preprocessing
# - Original fundus image loaded by OpenCV.
# - Resize to 300x300.
# - Convert BGR to RGB.
# - Apply fixed CLAHE to the green channel only, then merge RGB.
# - Cast to float32 for the existing EfficientNetB3 training pipeline.
#
# %%
def apply_fixed_green_channel_clahe(img_rgb: np.ndarray) -> np.ndarray:
    r, g, b = cv2.split(img_rgb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_clahe = clahe.apply(g)
    return cv2.merge([r, g_clahe, b])


def preprocess_cv2(filepath: str, cfg: ExperimentConfig = CFG) -> np.ndarray:
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Could not read image: {filepath}")

    img = cv2.resize(img, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if cfg.preprocessing_strategy == "green_clahe_only":
        img = apply_fixed_green_channel_clahe(img)
    else:
        raise ValueError(f"Unknown preprocessing_strategy: {cfg.preprocessing_strategy}")

    return img.astype(np.float32)


def parse_image(filepath, label):
    img = tf.py_function(
        func=lambda fp: preprocess_cv2(fp.numpy().decode("utf-8")),
        inp=[filepath],
        Tout=tf.float32,
    )
    img.set_shape([CFG.img_size, CFG.img_size, 3])
    label = tf.cast(label, tf.int32)
    return img, label


def augment_tensor(img, label):
    if CFG.augmentation_strategy == "none":
        return img, label
    if CFG.augmentation_strategy != "online_light":
        raise ValueError(f"Unknown augmentation_strategy: {CFG.augmentation_strategy}")

    img = tf.image.random_flip_left_right(img, seed=CFG.seed)
    img = tf.image.random_brightness(img, max_delta=15.0, seed=CFG.seed)
    img = tf.image.random_contrast(img, lower=0.90, upper=1.10, seed=CFG.seed)
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img, label


def make_dataset(df: pd.DataFrame, training: bool = False, batch_size: int = CFG.batch_size):
    filepaths = df["filepath"].values
    labels = df[CFG.label_col].values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=CFG.seed, reshuffle_each_iteration=True)
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training and CFG.augmentation_strategy != "none":
        ds = ds.map(augment_tensor, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds_phase1 = make_dataset(train_df, training=True, batch_size=CFG.phase1_batch_size)
train_ds_phase2 = make_dataset(train_df, training=True, batch_size=CFG.phase2_batch_size)
val_ds = make_dataset(val_df, training=False, batch_size=CFG.batch_size)
test_ds = make_dataset(test_df, training=False, batch_size=CFG.batch_size)

print(
    "Dataset batch sizes:",
    {
        "phase1_train": CFG.phase1_batch_size,
        "phase2_train": CFG.phase2_batch_size,
        "phase2_gradient_accumulation_steps": CFG.phase2_gradient_accumulation_steps,
        "phase2_effective_batch": CFG.phase2_batch_size * CFG.phase2_gradient_accumulation_steps,
        "val": CFG.batch_size,
        "test": CFG.batch_size,
    },
)

# %% [markdown]
# ## Imbalance handling
#
# This pipeline uses the full training set and class weights based only on
# the training split. It does not undersample class 0 and does not generate fixed
# offline augmented files. This isolates imbalance handling from augmentation.

# %%
def get_class_weights(train: pd.DataFrame, cfg: ExperimentConfig):
    if cfg.imbalance_strategy == "none":
        return None
    if cfg.imbalance_strategy != "class_weight":
        raise ValueError(f"Unknown imbalance_strategy: {cfg.imbalance_strategy}")

    classes = np.arange(cfg.num_classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train[cfg.label_col].values,
    )
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


class_weight_dict = get_class_weights(train_df, CFG)
print("Class weights:", class_weight_dict)

# %% [markdown]
# ## Model and fine-tuning

# %%
def build_model(cfg: ExperimentConfig = CFG):
    if cfg.backbone_name != "EfficientNetB3":
        raise ValueError(f"Unsupported backbone_name: {cfg.backbone_name}")

    base = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(cfg.img_size, cfg.img_size, 3),
    )
    base.trainable = False

    inputs = layers.Input(shape=(cfg.img_size, cfg.img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(cfg.num_classes, activation="softmax", name="grade_output")(x)
    return Model(inputs, outputs), base


def freeze_backbone_batchnorm(base_model: tf.keras.Model) -> None:
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


model, base_model = build_model(CFG)
model.summary()

# %%
def compile_model(
    model: tf.keras.Model,
    learning_rate: float,
    weight_decay: float,
    gradient_accumulation_steps: int | None = None,
) -> None:
    optimizer_kwargs = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
    }
    if gradient_accumulation_steps is not None and gradient_accumulation_steps > 1:
        optimizer_kwargs["gradient_accumulation_steps"] = gradient_accumulation_steps

    model.compile(
        optimizer=AdamW(**optimizer_kwargs),
        loss=CFG.loss_name,
        metrics=["accuracy"],
    )


class QWKCallback(tf.keras.callbacks.Callback):
    """Compute validation quadratic weighted kappa at epoch end."""

    def __init__(self, val_dataset):
        super().__init__()
        self.val_dataset = val_dataset

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_true, y_pred = [], []
        for images, labels in self.val_dataset:
            grade_probs = self.model.predict(images, verbose=0)
            y_pred.extend(np.argmax(grade_probs, axis=1))
            y_true.extend(labels.numpy())
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        logs["val_qwk"] = qwk
        print(f" â€” val_qwk: {qwk:.4f}")


compile_model(model, CFG.lr_phase1, CFG.weight_decay_phase1)

callbacks_p1 = [
    CSVLogger(str(OUTPUT_DIR / "phase1_epoch_history.csv"), append=False),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
    ModelCheckpoint(
        str(MODEL_DIR / "best_phase1.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
]

print("\n=== PHASE 1: Train classification head only ===")
training_start_time = time.perf_counter()
phase1_start_time = training_start_time
hist1 = model.fit(
    train_ds_phase1,
    validation_data=val_ds,
    epochs=CFG.phase1_epochs,
    callbacks=callbacks_p1,
    class_weight=class_weight_dict,
    verbose=1,
)
phase1_end_time = time.perf_counter()

# %%
base_model.trainable = True
if CFG.freeze_backbone_batchnorm:
    freeze_backbone_batchnorm(base_model)

compile_model(
    model,
    CFG.lr_phase2,
    CFG.weight_decay_phase2,
    gradient_accumulation_steps=CFG.phase2_gradient_accumulation_steps,
)

callbacks_p2 = [
    QWKCallback(val_ds),
    CSVLogger(str(OUTPUT_DIR / "phase2_epoch_history.csv"), append=False),
    ReduceLROnPlateau(
        monitor="val_qwk",
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
    ModelCheckpoint(
        str(MODEL_DIR / "best_phase2.keras"),
        monitor="val_qwk",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_qwk",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
]

print(
    "\n=== PHASE 2: Fine tune backbone "
    f"(physical batch={CFG.phase2_batch_size}, "
    f"gradient accumulation={CFG.phase2_gradient_accumulation_steps}, "
    f"effective batch={CFG.phase2_batch_size * CFG.phase2_gradient_accumulation_steps}) ==="
)
phase2_start_time = time.perf_counter()
hist2 = model.fit(
    train_ds_phase2,
    validation_data=val_ds,
    epochs=CFG.phase2_epochs,
    callbacks=callbacks_p2,
    class_weight=class_weight_dict,
    verbose=1,
)
phase2_end_time = time.perf_counter()

phase1_duration_seconds = phase1_end_time - phase1_start_time
phase2_duration_seconds = phase2_end_time - phase2_start_time
total_training_seconds = phase2_end_time - training_start_time
training_time = {
    "phase1_seconds": phase1_duration_seconds,
    "phase1_minutes": phase1_duration_seconds / 60,
    "phase1_hours": phase1_duration_seconds / 3600,
    "phase2_seconds": phase2_duration_seconds,
    "phase2_minutes": phase2_duration_seconds / 60,
    "phase2_hours": phase2_duration_seconds / 3600,
    "total_training_seconds": total_training_seconds,
    "total_training_minutes": total_training_seconds / 60,
    "total_training_hours": total_training_seconds / 3600,
}
with open(OUTPUT_DIR / "training_time.json", "w") as f:
    json.dump(training_time, f, indent=2)

print(
    "Total training time: "
    f"{total_training_seconds:.2f} seconds | "
    f"{total_training_seconds / 60:.2f} minutes | "
    f"{total_training_seconds / 3600:.2f} hours"
)
print(f"Phase 1 duration: {phase1_duration_seconds:.2f} seconds")
print(f"Phase 2 duration: {phase2_duration_seconds:.2f} seconds")
print(f"Saved training time summary: {OUTPUT_DIR / 'training_time.json'}")


def save_training_history_immediately(histories: dict, output_dir: Path) -> None:
    """Persist combined training history before loading/evaluating the checkpoint."""
    rows = []
    global_epoch = 0
    for phase_name, history_obj in histories.items():
        history_dict = history_obj.history
        phase_epochs = max(len(values) for values in history_dict.values()) if history_dict else 0
        for local_epoch in range(phase_epochs):
            global_epoch += 1
            row = {
                "phase": phase_name,
                "local_epoch": local_epoch + 1,
                "global_epoch": global_epoch,
            }
            for metric_name, values in history_dict.items():
                row[metric_name] = values[local_epoch] if local_epoch < len(values) else np.nan
            rows.append(row)

    history_df_immediate = pd.DataFrame(rows)
    history_df_immediate.to_csv(output_dir / "history.csv", index=False)

    def convert_history_value(value):
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    with open(output_dir / "history.json", "w") as f:
        json.dump(
            {phase: hist.history for phase, hist in histories.items()},
            f,
            indent=2,
            default=convert_history_value,
        )


history = {"phase1": hist1, "phase2": hist2}
save_training_history_immediately(history, OUTPUT_DIR)
print(f"Saved combined training history before evaluation: {OUTPUT_DIR / 'history.csv'}")

# %% [markdown]
# ## Evaluation
#
# Validation is used for model selection. Test is evaluated only after training
# and checkpoint selection are complete.

# %%
EVAL_DIR = OUTPUT_DIR / "evaluation"
PREDICTION_DIR = EVAL_DIR / "predictions"
REPORT_DIR = EVAL_DIR / "reports"
for directory in [EVAL_DIR, PREDICTION_DIR, REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "best_phase2.keras"
best_model = tf.keras.models.load_model(str(BEST_MODEL_PATH))
print(f"Loaded checkpoint for final evaluation: {BEST_MODEL_PATH}")


def to_serializable(value):
    """Convert NumPy scalar values to JSON-serializable Python scalars."""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=to_serializable)


def build_history_frame(histories: dict) -> pd.DataFrame:
    rows = []
    global_epoch = 0
    for phase_name, history_obj in histories.items():
        history_dict = history_obj.history
        phase_epochs = max(len(values) for values in history_dict.values())
        for local_epoch in range(phase_epochs):
            global_epoch += 1
            row = {
                "phase": phase_name,
                "local_epoch": local_epoch + 1,
                "global_epoch": global_epoch,
            }
            for metric_name, values in history_dict.items():
                row[metric_name] = values[local_epoch] if local_epoch < len(values) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def collect_predictions(model: tf.keras.Model, dataset):
    y_true, y_pred, y_prob = [], [], []
    for images, labels in dataset:
        grade_probs = model.predict(images, verbose=0)
        y_prob.append(grade_probs)
        y_pred.append(np.argmax(grade_probs, axis=1))
        y_true.append(labels.numpy())
    return (
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.concatenate(y_prob),
    )


def make_prediction_frame(
    split_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    pred_df = split_df[["image", "patient_id", "filepath", CFG.label_col]].copy()
    pred_df = pred_df.rename(columns={CFG.label_col: "true_label"})
    pred_df["pred_label"] = y_pred
    pred_df["true_class"] = [CLASS_NAMES[label] for label in y_true]
    pred_df["pred_class"] = [CLASS_NAMES[label] for label in y_pred]
    pred_df["correct"] = pred_df["true_label"] == pred_df["pred_label"]
    pred_df["referable_true"] = (pred_df["true_label"] >= 2).astype(int)
    pred_df["referable_prob"] = y_prob[:, 2:].sum(axis=1)
    pred_df["referable_pred"] = (pred_df["referable_prob"] >= 0.5).astype(int)

    for class_idx, class_name in enumerate(CLASS_NAMES):
        safe_name = class_name.lower().replace(" ", "_")
        pred_df[f"prob_{class_idx}_{safe_name}"] = y_prob[:, class_idx]

    return pred_df


def referable_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_true_ref = (y_true >= 2).astype(int)
    y_prob_ref = y_prob[:, 2:].sum(axis=1)
    y_pred_ref = (y_prob_ref >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_ref, y_pred_ref, labels=[0, 1]).ravel()

    metrics = {
        "referable_accuracy": accuracy_score(y_true_ref, y_pred_ref),
        "referable_sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "referable_specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "referable_tp": int(tp),
        "referable_fp": int(fp),
        "referable_tn": int(tn),
        "referable_fn": int(fn),
    }
    try:
        metrics["referable_auc"] = roc_auc_score(y_true_ref, y_prob_ref)
    except ValueError:
        metrics["referable_auc"] = np.nan
    return metrics


def evaluate_split(
    model: tf.keras.Model,
    dataset,
    split_df: pd.DataFrame,
    split_name: str,
) -> dict:
    y_true, y_pred, y_prob = collect_predictions(model, dataset)
    pred_df = make_prediction_frame(split_df, y_true, y_pred, y_prob)
    pred_df.to_csv(PREDICTION_DIR / f"{split_name}_predictions.csv", index=False)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(CFG.num_classes)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(CFG.num_classes)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    metrics = {
        "split": split_name,
        "n_images": int(len(y_true)),
        "n_patients": int(split_df["patient_id"].nunique()),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "mild_precision": report_dict["Mild"]["precision"],
        "mild_recall": report_dict["Mild"]["recall"],
        "mild_f1": report_dict["Mild"]["f1-score"],
        "no_dr_recall": report_dict["No DR"]["recall"],
        **referable_metrics(y_true, y_prob),
    }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(CFG.num_classes)))
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        REPORT_DIR / f"{split_name}_confusion_matrix.csv"
    )
    pd.DataFrame(report_dict).transpose().to_csv(
        REPORT_DIR / f"{split_name}_classification_report.csv"
    )
    save_json(metrics, REPORT_DIR / f"{split_name}_metrics.json")
    with open(REPORT_DIR / f"{split_name}_classification_report.txt", "w") as f:
        f.write(report_text)

    print(f"\n=== {split_name.upper()} ===")
    print(json.dumps(metrics, indent=2, default=to_serializable))
    print(report_text)

    return {
        "split": split_name,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
        "report_text": report_text,
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path,
    normalize: bool = False,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(CFG.num_classes)))
    values = cm.astype(float)
    fmt = "d"
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)
        fmt = ".2f"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        values if normalize else cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        vmin=0 if normalize else None,
        vmax=1 if normalize else None,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


def plot_training_curves(history_df: pd.DataFrame, save_path: Path) -> None:
    metric_pairs = [
        ("loss", "val_loss", "Loss"),
        ("accuracy", "val_accuracy", "5-class DR accuracy"),
        ("lr", None, "Learning rate"),
        ("val_qwk", None, "Validation QWK"),
    ]
    available = [
        pair for pair in metric_pairs if pair[0] in history_df.columns or pair[1] in history_df.columns
    ]

    fig, axes = plt.subplots(len(available), 1, figsize=(9, 4 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]

    for ax, (train_metric, val_metric, title) in zip(axes, available):
        if train_metric in history_df.columns:
            ax.plot(history_df["global_epoch"], history_df[train_metric], marker="o", label=train_metric)
        if val_metric and val_metric in history_df.columns:
            ax.plot(history_df["global_epoch"], history_df[val_metric], marker="o", label=val_metric)
        ax.set_title(title)
        ax.set_xlabel("Global epoch")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


def compute_grade_metrics(
    split_name: str,
    split_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(CFG.num_classes)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    metrics = {
        "split": split_name,
        "n_images": int(len(y_true)),
        "n_patients": int(split_df["patient_id"].nunique()),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "mild_precision": report_dict["Mild"]["precision"],
        "mild_recall": report_dict["Mild"]["recall"],
        "mild_f1": report_dict["Mild"]["f1-score"],
        "no_dr_recall": report_dict["No DR"]["recall"],
        **referable_metrics(y_true, y_prob),
    }
    return metrics


history = {"phase1": hist1, "phase2": hist2}
history_df = build_history_frame(history)
history_df.to_csv(OUTPUT_DIR / "history.csv", index=False)
save_json({phase: hist.history for phase, hist in history.items()}, OUTPUT_DIR / "history.json")

val_results = evaluate_split(best_model, val_ds, val_df, "val")
test_results = evaluate_split(best_model, test_ds, test_df, "test")

summary_df = pd.DataFrame([val_results["metrics"], test_results["metrics"]])
summary_df.to_csv(EVAL_DIR / "metrics_summary.csv", index=False)

save_json(
    {
        "description": "5-Class DR Classification - No Mild Threshold Calibration",
        "config": asdict(CFG),
        "best_model_path": str(BEST_MODEL_PATH),
        "metrics": {
            "val": val_results["metrics"],
            "test": test_results["metrics"],
        },
    },
    EVAL_DIR / "evaluation_summary.json",
)

with open(OUTPUT_DIR / "results_corrected_pipeline.txt", "w") as f:
    f.write("=== 5-Class DR Classification - No Mild Threshold Calibration ===\n")
    f.write(f"Best model checkpoint: {BEST_MODEL_PATH}\n\n")
    f.write("Configuration:\n")
    f.write(json.dumps(asdict(CFG), indent=2))
    for results in [val_results, test_results]:
        f.write(f"\n\n=== {results['split'].upper()} METRICS ===\n")
        f.write(json.dumps(results["metrics"], indent=2, default=to_serializable))
        f.write(f"\n\n=== {results['split'].upper()} CLASSIFICATION REPORT ===\n")
        f.write(results["report_text"])

plot_training_curves(history_df, PLOT_DIR / "training_curves.png")
for results in [val_results, test_results]:
    split_name = results["split"]
    plot_confusion_matrix(
        results["y_true"],
        results["y_pred"],
        f"{split_name.upper()} confusion matrix",
        PLOT_DIR / f"cm_{split_name}.png",
        normalize=False,
    )
    plot_confusion_matrix(
        results["y_true"],
        results["y_pred"],
        f"{split_name.upper()} normalized confusion matrix",
        PLOT_DIR / f"cm_{split_name}_normalized.png",
        normalize=True,
    )

print(f"Saved original-only evaluation artifacts under: {EVAL_DIR}")
print(f"Saved plots under: {PLOT_DIR}")
