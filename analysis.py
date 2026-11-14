from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use("Agg")  # Save charts without opening graphics windows.

import matplotlib.pyplot as plt
import medmnist
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)

# Project settings
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"

IMAGE_SIZE = 64
BATCH_SIZE = 64
SEED = 42


def main():
    # Create folders for downloaded data and generated results.
    for folder in (DATA_DIR, OUTPUT_DIR, MODEL_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(SEED)

    # STEP 1: CHECK THE ENVIRONMENT
    print("Python version:", platform.python_version())
    print("Python executable:", sys.executable)
    print("TensorFlow version:", tf.__version__)
    print("MedMNIST version:", medmnist.__version__)
    print("NumPy version:", np.__version__)
    print("Pandas version:", pd.__version__)
    print("Matplotlib version:", matplotlib.__version__)
    print("Seaborn version:", sns.__version__)
    print("Scikit-learn version:", sklearn.__version__)

    print("\nAvailable GPUs:", tf.config.list_physical_devices("GPU"))

    result = tf.reduce_sum(tf.constant([1.0, 2.0, 3.0]))
    print("TensorFlow calculation:", result.numpy())

    print("\nStep 1 completed successfully.")
    # STEP 2: DOWNLOAD AND INSPECT ORGANAMNIST
    info = medmnist.INFO["organamnist"]

    class_names = [
        info["label"][str(i)]
        for i in range(len(info["label"]))
    ]
    num_classes = len(class_names)

    # Keep the official splits.
    datasets = {}

    for split in ("train", "val", "test"):
        print(f"\nLoading {split} split...", flush=True)

        datasets[split] = medmnist.OrganAMNIST(
            split=split,
            root=str(DATA_DIR),
            size=IMAGE_SIZE,
            download=True,
        )

    print("\nDATASET SUMMARY")
    print("Number of classes:", num_classes)

    count_rows = []

    for split, dataset in datasets.items():
        images = dataset.imgs
        labels = dataset.labels.reshape(-1).astype("int32")

        assert len(images) == len(labels)
        assert images.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE)
        assert np.all((labels >= 0) & (labels < num_classes))

        print(
            f"{split:5s}: {len(images):,} images | "
            f"shape: {images.shape} | dtype: {images.dtype}"
        )

        counts = np.bincount(labels, minlength=num_classes)

        for class_id, class_name in enumerate(class_names):
            count_rows.append({
                "split": split,
                "class_id": class_id,
                "class_name": class_name,
                "image_count": int(counts[class_id]),
            })

    # Save image counts for the report.
    class_counts = pd.DataFrame(count_rows)
    class_counts.to_csv(
        OUTPUT_DIR / "class_counts.csv",
        index=False,
    )

    print("\nIMAGE COUNTS BY CLASS")
    print(
        class_counts.pivot(
            index="class_name",
            columns="split",
            values="image_count",
        ).to_string()
    )

    # Explore training images only.
    train_images = datasets["train"].imgs
    train_labels = datasets["train"].labels.reshape(-1)

    print(
        "\nTraining pixel range:",
        int(train_images.min()),
        "to",
        int(train_images.max()),
    )

    # Save one example from each category.
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.ravel()

    for class_id, class_name in enumerate(class_names):
        candidates = np.flatnonzero(train_labels == class_id)

        if candidates.size:
            image_index = rng.choice(candidates)
            axes[class_id].imshow(
                train_images[image_index],
                cmap="gray",
                vmin=0,
                vmax=255,
            )

        axes[class_id].set_title(class_name.replace("-", " "))
        axes[class_id].axis("off")

    for ax in axes[num_classes:]:
        ax.axis("off")

    fig.suptitle("OrganAMNIST: Training Examples", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "training_examples.png", dpi=200)
    plt.close(fig)

    # Save the training class distribution.
    training_counts = (
        class_counts[class_counts["split"] == "train"]
        .sort_values("class_id")
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        training_counts["class_name"],
        training_counts["image_count"],
        color="steelblue",
    )
    ax.invert_yaxis()
    ax.set_xlabel("Number of training images")
    ax.set_title("OrganAMNIST: Training Class Distribution")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "training_class_distribution.png",
        dpi=200,
    )
    plt.close(fig)

    print("\nOutput folder:", OUTPUT_DIR)
    print("Step 2 completed successfully.")
    # STEP 3: PREPROCESSING AND DATA AUGMENTATION

    # Small changes to training images.
    # Avoid flips because our labels distinguish left and right structures.
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(
                factor=0.015,  # Approximately +/- 5.4 degrees.
                fill_mode="nearest",
                seed=SEED,
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=0.03,
                width_factor=0.03,
                fill_mode="nearest",
                seed=SEED + 1,
            ),
        ],
        name="training_augmentation",
    )

    def normalize(images, labels):
        # Convert pixel values from 0–255 to 0–1.
        images = tf.cast(images, tf.float32) / 255.0
        return images, labels

    def augment(images, labels):
        images = augmentation(images, training=True)
        return images, labels

    prepared = {}

    for split, dataset in datasets.items():
        # Shape changes from (N, 64, 64) to (N, 64, 64, 1).
        images = dataset.imgs[..., np.newaxis]

        # Integer labels are suitable for sparse categorical cross-entropy.
        labels = dataset.labels.reshape(-1).astype("int32")

        ds = tf.data.Dataset.from_tensor_slices((images, labels))

        # Shuffle training images each epoch.
        if split == "train":
            ds = ds.shuffle(
                buffer_size=min(len(images), 5000),
                seed=SEED,
                reshuffle_each_iteration=True,
            )

        ds = ds.batch(BATCH_SIZE)

        # Normalize all three splits.
        ds = ds.map(
            normalize,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Apply random transformations only to training images.
        if split == "train":
            ds = ds.map(
                augment,
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        prepared[split] = ds.prefetch(tf.data.AUTOTUNE)

    train_ds = prepared["train"]
    val_ds = prepared["val"]
    test_ds = prepared["test"]

    # Inspect one training batch.
    image_batch, label_batch = next(iter(train_ds))

    print("\nPREPROCESSING SUMMARY")
    print("Image batch shape:", tuple(image_batch.shape))
    print("Label batch shape:", tuple(label_batch.shape))
    print("Image dtype:", image_batch.dtype.name)
    print("Label dtype:", label_batch.dtype.name)

    print(
        "Pixel range:",
        f"{float(tf.reduce_min(image_batch).numpy()):.4f}",
        "to",
        f"{float(tf.reduce_max(image_batch).numpy()):.4f}",
    )

    print("Training batches:", int(train_ds.cardinality().numpy()))
    print("Validation batches:", int(val_ds.cardinality().numpy()))
    print("Test batches:", int(test_ds.cardinality().numpy()))

    print("\nStep 3 completed successfully.")

    # STEP 4: BUILD AND TRAIN THE CNN BASELINE
    cnn_model_path = MODEL_DIR / "cnn_baseline_best.keras"
    cnn_history_path = OUTPUT_DIR / "cnn_baseline_history.csv"

    # Reuse completed training when running later assignment steps.
    if cnn_model_path.exists() and cnn_history_path.exists():
        print("\nLoading the saved CNN baseline...")

        cnn_model = tf.keras.models.load_model(str(cnn_model_path))
        cnn_history = pd.read_csv(cnn_history_path)

    else:
        tf.keras.utils.set_random_seed(SEED)

        cnn_model = tf.keras.Sequential(
            [
                tf.keras.Input(
                    shape=(IMAGE_SIZE, IMAGE_SIZE, 1)
                ),

                # First convolutional block.
                tf.keras.layers.Conv2D(
                    32,
                    kernel_size=3,
                    padding="same",
                    use_bias=False,
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.MaxPooling2D(pool_size=2),

                # Second convolutional block.
                tf.keras.layers.Conv2D(
                    64,
                    kernel_size=3,
                    padding="same",
                    use_bias=False,
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.MaxPooling2D(pool_size=2),

                # Classification layers.
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(
                    num_classes,
                    activation="softmax",
                ),
            ],
            name="cnn_baseline",
        )

        cnn_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(
                    name="accuracy"
                )
            ],
        )

        cnn_model.summary()

        cnn_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(cnn_model_path),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
        ]

        print("\nTraining the CNN baseline...")

        history = cnn_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=20,
            callbacks=cnn_callbacks,
            verbose=2,
        )

        # Save results from every completed epoch.
        cnn_history = pd.DataFrame(history.history)
        cnn_history.insert(
            0,
            "epoch",
            np.arange(1, len(cnn_history) + 1),
        )
        cnn_history.to_csv(cnn_history_path, index=False)

        # Load the checkpoint selected by lowest validation loss.
        cnn_model = tf.keras.models.load_model(
            str(cnn_model_path)
        )

    # Save the architecture summary for the report.
    summary_lines = []

    def record_cnn_summary(line, **kwargs):
        summary_lines.append(line)

    cnn_model.summary(print_fn=record_cnn_summary)

    (OUTPUT_DIR / "cnn_baseline_summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    # Plot training and validation performance.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        cnn_history["epoch"],
        cnn_history["accuracy"],
        label="Training",
    )
    axes[0].plot(
        cnn_history["epoch"],
        cnn_history["val_accuracy"],
        label="Validation",
    )
    axes[0].set_title("CNN Baseline: Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(
        cnn_history["epoch"],
        cnn_history["loss"],
        label="Training",
    )
    axes[1].plot(
        cnn_history["epoch"],
        cnn_history["val_loss"],
        label="Validation",
    )
    axes[1].set_title("CNN Baseline: Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "cnn_baseline_training_curves.png",
        dpi=200,
    )
    plt.close(fig)

    # Evaluate the selected model on validation data.
    # Reserve test data for final evaluation after tuning.
    cnn_val_results = cnn_model.evaluate(
        val_ds,
        verbose=0,
        return_dict=True,
    )

    pd.DataFrame([cnn_val_results]).to_csv(
        OUTPUT_DIR / "cnn_baseline_validation.csv",
        index=False,
    )

    best_row = cnn_history.loc[
        cnn_history["val_loss"].idxmin()
    ]

    print("\nCNN BASELINE RESULTS")
    print("Epochs completed:", len(cnn_history))
    print("Selected epoch:", int(best_row["epoch"]))
    print(
        "Validation accuracy:",
        f"{cnn_val_results['accuracy']:.2%}",
    )
    print(
        "Validation loss:",
        f"{cnn_val_results['loss']:.4f}",
    )
    print("Best model saved to:", cnn_model_path)
    print("\nStep 4 completed successfully.")

    # STEP 5: BUILD AND TRAIN THE CNN–LSTM MODEL

    # This function can also be reused for later tuning experiments.
    def build_cnn_lstm(
        lstm_units=64,
        dropout_rate=0.3,
        learning_rate=0.001,
    ):
        sequence_steps = IMAGE_SIZE // 4
        features_per_step = sequence_steps * 64

        model = tf.keras.Sequential(
            [
                tf.keras.Input(
                    shape=(IMAGE_SIZE, IMAGE_SIZE, 1)
                ),

                # Same convolutional design as the CNN baseline.
                tf.keras.layers.Conv2D(
                    32,
                    kernel_size=3,
                    padding="same",
                    use_bias=False,
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.MaxPooling2D(pool_size=2),

                tf.keras.layers.Conv2D(
                    64,
                    kernel_size=3,
                    padding="same",
                    use_bias=False,
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.MaxPooling2D(pool_size=2),

                # (16, 16, 64) becomes (16, 1024).
                # Each sequence step represents one feature-map row.
                tf.keras.layers.Reshape(
                    (sequence_steps, features_per_step),
                    name="spatial_sequence",
                ),

                # Summarize the ordered spatial features.
                tf.keras.layers.LSTM(
                    lstm_units,
                    return_sequences=False,
                    name="spatial_lstm",
                ),
                tf.keras.layers.Dropout(dropout_rate),

                tf.keras.layers.Dense(
                    num_classes,
                    activation="softmax",
                ),
            ],
            name="cnn_lstm",
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(
                    name="accuracy"
                )
            ],
        )

        return model

    cnn_lstm_model_path = MODEL_DIR / "cnn_lstm_initial_best.keras"
    cnn_lstm_history_path = OUTPUT_DIR / "cnn_lstm_initial_history.csv"

    if (
        cnn_lstm_model_path.exists()
        and cnn_lstm_history_path.exists()
    ):
        print("\nLoading the saved CNN–LSTM model...")

        cnn_lstm_model = tf.keras.models.load_model(
            str(cnn_lstm_model_path)
        )
        cnn_lstm_history = pd.read_csv(cnn_lstm_history_path)

    else:
        tf.keras.utils.set_random_seed(SEED)

        cnn_lstm_model = build_cnn_lstm(
            lstm_units=64,
            dropout_rate=0.3,
            learning_rate=0.001,
        )

        cnn_lstm_model.summary()

        cnn_lstm_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(cnn_lstm_model_path),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
        ]

        print("\nTraining the CNN–LSTM model...")

        cnn_lstm_fit = cnn_lstm_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=20,
            callbacks=cnn_lstm_callbacks,
            verbose=2,
        )

        cnn_lstm_history = pd.DataFrame(cnn_lstm_fit.history)
        cnn_lstm_history.insert(
            0,
            "epoch",
            np.arange(1, len(cnn_lstm_history) + 1),
        )
        cnn_lstm_history.to_csv(
            cnn_lstm_history_path,
            index=False,
        )

        # Use the model selected by lowest validation loss.
        cnn_lstm_model = tf.keras.models.load_model(
            str(cnn_lstm_model_path)
        )

    # Save the architecture summary.
    cnn_lstm_summary_lines = []

    def record_cnn_lstm_summary(line, **kwargs):
        cnn_lstm_summary_lines.append(line)

    cnn_lstm_model.summary(
        print_fn=record_cnn_lstm_summary
    )

    (OUTPUT_DIR / "cnn_lstm_initial_summary.txt").write_text(
        "\n".join(cnn_lstm_summary_lines),
        encoding="utf-8",
    )

    # Save training and validation curves.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric, title in (
        (axes[0], "accuracy", "Accuracy"),
        (axes[1], "loss", "Loss"),
    ):
        ax.plot(
            cnn_lstm_history["epoch"],
            cnn_lstm_history[metric],
            label="Training",
        )
        ax.plot(
            cnn_lstm_history["epoch"],
            cnn_lstm_history[f"val_{metric}"],
            label="Validation",
        )
        ax.set_title(f"CNN–LSTM: {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "cnn_lstm_initial_training_curves.png",
        dpi=200,
    )
    plt.close(fig)

    # Evaluate on validation data; keep test data for final evaluation.
    cnn_lstm_val_results = cnn_lstm_model.evaluate(
        val_ds,
        verbose=0,
        return_dict=True,
    )

    pd.DataFrame([cnn_lstm_val_results]).to_csv(
        OUTPUT_DIR / "cnn_lstm_initial_validation.csv",
        index=False,
    )

    # Compare both models at their selected checkpoints.
    validation_comparison = pd.DataFrame([
        {
            "model": "CNN baseline",
            "model_parameters": cnn_model.count_params(),
            "validation_accuracy": cnn_val_results["accuracy"],
            "validation_loss": cnn_val_results["loss"],
        },
        {
            "model": "Initial CNN-LSTM",
            "model_parameters": cnn_lstm_model.count_params(),
            "validation_accuracy": cnn_lstm_val_results["accuracy"],
            "validation_loss": cnn_lstm_val_results["loss"],
        },
    ])

    validation_comparison.to_csv(
        OUTPUT_DIR / "initial_validation_comparison.csv",
        index=False,
    )

    cnn_lstm_best_row = cnn_lstm_history.loc[
        cnn_lstm_history["val_loss"].idxmin()
    ]

    print("\nCNN–LSTM RESULTS")
    print("Epochs completed:", len(cnn_lstm_history))
    print("Selected epoch:", int(cnn_lstm_best_row["epoch"]))
    print(
        "Validation accuracy:",
        f"{cnn_lstm_val_results['accuracy']:.2%}",
    )
    print(
        "Validation loss:",
        f"{cnn_lstm_val_results['loss']:.4f}",
    )

    print("\nINITIAL VALIDATION COMPARISON")
    print(validation_comparison.to_string(index=False))

    print("\nStep 5 completed successfully.")
    # STEP 6: CNN–LSTM HYPERPARAMETER EXPERIMENTS

    # Recreate the training pipeline for each new experiment.
    # This resets its shuffle and augmentation seeds.
    def make_trial_training_data():
        trial_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(
                factor=0.015,
                fill_mode="nearest",
                seed=SEED,
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=0.03,
                width_factor=0.03,
                fill_mode="nearest",
                seed=SEED + 1,
            ),
        ])

        def augment_trial(images, labels):
            return trial_augmentation(images, training=True), labels

        images = datasets["train"].imgs[..., np.newaxis]
        labels = (
            datasets["train"].labels.reshape(-1).astype("int32")
        )

        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        ds = ds.shuffle(
            buffer_size=min(len(images), 5000),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
        ds = ds.batch(BATCH_SIZE)
        ds = ds.map(normalize, num_parallel_calls=1)
        ds = ds.map(augment_trial, num_parallel_calls=1)

        return ds.prefetch(1)

    # Include the original model: tuning may not improve on it.
    tuning_rows = [{
        "experiment": "initial",
        "lstm_units": 64,
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "model_parameters": cnn_lstm_model.count_params(),
        "epochs_completed": len(cnn_lstm_history),
        "best_epoch": int(cnn_lstm_best_row["epoch"]),
        "validation_accuracy": cnn_lstm_val_results["accuracy"],
        "validation_loss": cnn_lstm_val_results["loss"],
        "model_path": str(cnn_lstm_model_path),
    }]

    experiments = [
        {
            "name": "lower_learning_rate",
            "lstm_units": 64,
            "dropout_rate": 0.3,
            "learning_rate": 0.0003,
        },
        {
            "name": "larger_lstm",
            "lstm_units": 128,
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
        },
        {
            "name": "higher_dropout",
            "lstm_units": 64,
            "dropout_rate": 0.5,
            "learning_rate": 0.001,
        },
    ]

    for experiment in experiments:
        experiment_name = experiment["name"]
        print(f"\nEXPERIMENT: {experiment_name}", flush=True)

        trial_model_path = (
            MODEL_DIR / f"cnn_lstm_{experiment_name}_best.keras"
        )
        trial_history_path = (
            OUTPUT_DIR / f"cnn_lstm_{experiment_name}_history.csv"
        )

        # Reuse completed experiments on subsequent script runs.
        if trial_model_path.exists() and trial_history_path.exists():
            print("Loading completed experiment...")

            trial_model = tf.keras.models.load_model(
                str(trial_model_path)
            )
            trial_history = pd.read_csv(trial_history_path)

        else:
            tf.keras.utils.set_random_seed(SEED)
            trial_train_ds = make_trial_training_data()

            # Build fresh weights for each configuration.
            trial_model = build_cnn_lstm(
                lstm_units=experiment["lstm_units"],
                dropout_rate=experiment["dropout_rate"],
                learning_rate=experiment["learning_rate"],
            )

            trial_callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(trial_model_path),
                    monitor="val_loss",
                    mode="min",
                    save_best_only=True,
                    verbose=1,
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=5,
                    restore_best_weights=True,
                    verbose=1,
                ),
            ]

            trial_fit = trial_model.fit(
                trial_train_ds,
                validation_data=val_ds,
                epochs=20,
                callbacks=trial_callbacks,
                verbose=2,
            )

            trial_history = pd.DataFrame(trial_fit.history)
            trial_history.insert(
                0,
                "epoch",
                np.arange(1, len(trial_history) + 1),
            )
            trial_history.to_csv(trial_history_path, index=False)

            trial_model = tf.keras.models.load_model(
                str(trial_model_path)
            )

            del trial_train_ds

        # Evaluate the best checkpoint on validation data only.
        trial_results = trial_model.evaluate(
            val_ds,
            verbose=0,
            return_dict=True,
        )

        trial_best_row = trial_history.loc[
            trial_history["val_loss"].idxmin()
        ]

        tuning_rows.append({
            "experiment": experiment_name,
            "lstm_units": experiment["lstm_units"],
            "dropout_rate": experiment["dropout_rate"],
            "learning_rate": experiment["learning_rate"],
            "model_parameters": trial_model.count_params(),
            "epochs_completed": len(trial_history),
            "best_epoch": int(trial_best_row["epoch"]),
            "validation_accuracy": trial_results["accuracy"],
            "validation_loss": trial_results["loss"],
            "model_path": str(trial_model_path),
        })

        # Save learning curves for this experiment.
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for ax, metric, title in (
            (axes[0], "accuracy", "Accuracy"),
            (axes[1], "loss", "Loss"),
        ):
            ax.plot(
                trial_history["epoch"],
                trial_history[metric],
                label="Training",
            )
            ax.plot(
                trial_history["epoch"],
                trial_history[f"val_{metric}"],
                label="Validation",
            )
            ax.set_title(f"{experiment_name}: {title}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.legend()

        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"cnn_lstm_{experiment_name}_curves.png",
            dpi=200,
        )
        plt.close(fig)

        print(
            f"Validation accuracy: {trial_results['accuracy']:.2%}"
        )
        print(
            f"Validation loss: {trial_results['loss']:.4f}"
        )

        del trial_model

    # Rank all configurations by validation loss.
    tuning_results = (
        pd.DataFrame(tuning_rows)
        .sort_values("validation_loss", kind="stable")
        .reset_index(drop=True)
    )

    tuning_results.to_csv(
        OUTPUT_DIR / "cnn_lstm_tuning_results.csv",
        index=False,
    )

    selected_configuration = tuning_results.iloc[0]

    selected_cnn_lstm = tf.keras.models.load_model(
        selected_configuration["model_path"]
    )
    selected_cnn_lstm.save(
        str(MODEL_DIR / "cnn_lstm_selected.keras")
    )

    # Save the selected settings for the report.
    tuning_results.iloc[[0]].to_csv(
        OUTPUT_DIR / "cnn_lstm_selected_configuration.csv",
        index=False,
    )

    print("\nTUNING RESULTS — LOWEST VALIDATION LOSS FIRST")
    print(
        tuning_results.drop(columns=["model_path"]).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nSELECTED CONFIGURATION")
    print("Experiment:", selected_configuration["experiment"])
    print("LSTM units:", int(selected_configuration["lstm_units"]))
    print("Dropout:", selected_configuration["dropout_rate"])
    print("Learning rate:", selected_configuration["learning_rate"])
    print(
        "Validation accuracy:",
        f"{selected_configuration['validation_accuracy']:.2%}",
    )
    print(
        "Validation loss:",
        f"{selected_configuration['validation_loss']:.4f}",
    )

    print("\nStep 6 completed successfully.")

    # STEP 7: FINAL TEST EVALUATION

    class_ids = np.arange(num_classes)

    # Read labels in the same order used by the test pipeline.
    # Step 3 keeps this pipeline unshuffled and unaugmented.
    y_test = np.concatenate([
        labels.numpy()
        for _, labels in test_ds
    ]).astype("int32")

    original_test_labels = (
        datasets["test"].labels.reshape(-1).astype("int32")
    )

    assert np.array_equal(y_test, original_test_labels)

    models_to_evaluate = [
        ("cnn_baseline", "CNN baseline", cnn_model),
        ("cnn_lstm_initial", "Initial CNN-LSTM", cnn_lstm_model),
        ("cnn_lstm_selected", "Selected CNN-LSTM", selected_cnn_lstm),
    ]

    test_rows = []

    for file_prefix, model_name, model in models_to_evaluate:
        print(f"\nTEST EVALUATION: {model_name}", flush=True)

        # Images are already normalized by Step 3.
        probabilities = model.predict(test_ds, verbose=1)

        assert probabilities.shape == (len(y_test), num_classes)
        assert np.all(np.isfinite(probabilities))
        assert np.allclose(
            probabilities.sum(axis=1),
            1.0,
            atol=1e-5,
        )

        # Choose the category with the highest probability.
        y_pred = np.argmax(
            probabilities,
            axis=1,
        ).astype("int32")

        report = classification_report(
            y_test,
            y_pred,
            labels=class_ids,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        test_accuracy = accuracy_score(y_test, y_pred)
        test_loss = log_loss(
            y_test,
            probabilities,
            labels=class_ids,
        )

        test_rows.append({
            "model": model_name,
            "model_parameters": model.count_params(),
            "test_images": len(y_test),
            "correct_predictions": int(np.sum(y_test == y_pred)),
            "incorrect_predictions": int(np.sum(y_test != y_pred)),
            "test_accuracy": test_accuracy,
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "test_log_loss": test_loss,
        })

        # Save per-class results and overall averages.
        pd.DataFrame(report).transpose().to_csv(
            OUTPUT_DIR / f"{file_prefix}_test_classification_report.csv"
        )

        report_text = classification_report(
            y_test,
            y_pred,
            labels=class_ids,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )

        (OUTPUT_DIR / f"{file_prefix}_test_report.txt").write_text(
            report_text,
            encoding="utf-8",
        )

        # Save individual predictions for later error analysis.
        prediction_table = pd.DataFrame({
            "test_image_index": np.arange(len(y_test)),
            "actual_class_id": y_test,
            "actual_class": [
                class_names[int(i)] for i in y_test
            ],
            "predicted_class_id": y_pred,
            "predicted_class": [
                class_names[int(i)] for i in y_pred
            ],
            "predicted_probability": probabilities.max(axis=1),
            "correct": y_test == y_pred,
        })

        prediction_table.to_csv(
            OUTPUT_DIR / f"{file_prefix}_test_predictions.csv",
            index=False,
        )

        # Preserve all category probabilities.
        np.savez_compressed(
            OUTPUT_DIR / f"{file_prefix}_test_probabilities.npz",
            y_true=y_test,
            y_pred=y_pred,
            probabilities=probabilities,
        )

        # Confusion matrix: actual categories are rows;
        # predicted categories are columns.
        cm_counts = confusion_matrix(
            y_test,
            y_pred,
            labels=class_ids,
        )

        cm_normalized = confusion_matrix(
            y_test,
            y_pred,
            labels=class_ids,
            normalize="true",
        )

        pd.DataFrame(
            cm_counts,
            index=class_names,
            columns=class_names,
        ).rename_axis("actual_class").to_csv(
            OUTPUT_DIR / f"{file_prefix}_test_confusion_counts.csv"
        )

        pd.DataFrame(
            cm_normalized,
            index=class_names,
            columns=class_names,
        ).rename_axis("actual_class").to_csv(
            OUTPUT_DIR / f"{file_prefix}_test_confusion_normalized.csv"
        )

        fig, axes = plt.subplots(1, 2, figsize=(21, 8))

        sns.heatmap(
            cm_counts,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[0],
        )
        axes[0].set_title(f"{model_name}: Test Image Counts")

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[1],
        )
        axes[1].set_title(
            f"{model_name}: Proportion Within Each Actual Class"
        )

        for ax in axes:
            ax.set_xlabel("Predicted category")
            ax.set_ylabel("Actual category")
            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=45,
                ha="right",
            )
            ax.set_yticklabels(
                ax.get_yticklabels(),
                rotation=0,
            )

        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"{file_prefix}_test_confusion_matrices.png",
            dpi=180,
        )
        plt.close(fig)

        print(f"\nTest accuracy: {test_accuracy:.2%}")
        print(f"Macro precision: {report['macro avg']['precision']:.4f}")
        print(f"Macro recall: {report['macro avg']['recall']:.4f}")
        print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")

    # Preserve model order: selection was completed using validation data.
    test_comparison = pd.DataFrame(test_rows)

    test_comparison.to_csv(
        OUTPUT_DIR / "final_test_comparison.csv",
        index=False,
    )

    print("\nFINAL TEST COMPARISON")
    print(
        test_comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nFiles saved to:", OUTPUT_DIR)
    print("Step 7 completed successfully.")
if __name__ == "__main__":
    main()