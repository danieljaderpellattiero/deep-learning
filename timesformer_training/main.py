from tqdm import tqdm
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoImageProcessor, TimesformerForVideoClassification, TimesformerConfig,
                          get_cosine_schedule_with_warmup)

import os
import glob
import torch
import evaluate
import numpy as np
import matplotlib.pyplot as plt

VIDEO_ROOT_DIR = "/home/elicer/yt-8m"
TRAIN_DIR = os.path.join(VIDEO_ROOT_DIR, "train")
VAL_DIR = os.path.join(VIDEO_ROOT_DIR, "validation")
PRETRAINED_MODEL_NAME = "facebook/timesformer-base-finetuned-k400"

NUM_CLASSES = 10
BATCH_SIZE = 8
NUM_EPOCHS = 10  # Consider increasing for scheduler to have more effect
LEARNING_RATE = 1e-4  # Initial learning rate
WARMUP_PROPORTION = 0.1  # Proportion of total training steps for linear warmup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOT_FILENAME = "training_metrics.png"
NUM_WORKERS = 4  # Number of workers for DataLoader

# --- Video Dataset Class ---
class VideoClassificationDataset(Dataset):
    def __init__(self, video_dir, image_processor, num_frames, class_to_idx=None):
        self.video_dir = video_dir
        self.image_processor = image_processor
        self.num_frames = num_frames
        self.video_files = []
        self.labels = []

        if class_to_idx is None:
            self.class_names = sorted(
                [d.name for d in os.scandir(video_dir) if d.is_dir()]
            )
            self.class_to_idx = {
                cls_name: i for i, cls_name in enumerate(self.class_names)
            }
        else:
            self.class_to_idx = class_to_idx
            self.class_names = sorted(list(class_to_idx.keys()))

        for class_name, label_idx in self.class_to_idx.items():
            class_path = os.path.join(video_dir, class_name)
            for video_file in glob.glob(os.path.join(class_path, "*.mp4")):
                self.video_files.append(video_file)
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.video_files)

    def _sample_frames(self, video_path):
        try:
            dummy_frame_height, dummy_frame_width = 224, 224
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            if total_frames == 0:
                vr = VideoReader(video_path, ctx=cpu(0))  # Retry
                total_frames = len(vr)
                if total_frames == 0:
                    print(
                        f"Warning: Video {video_path} has 0 frames even after retry. Returning dummy frames."
                    )
                    return [
                        np.zeros(
                            (dummy_frame_height, dummy_frame_width, 3), dtype=np.uint8
                        )
                        for _ in range(self.num_frames)
                    ]

            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = vr.get_batch(indices).asnumpy()
            if (
                frames.size == 0
                or frames.ndim != 4
                or frames.shape[3] != 3
                or frames.shape[0] != self.num_frames
            ):
                print(
                    f"Warning: Video {video_path} yielded problematic frames (shape: {frames.shape}, expected NF={self.num_frames}). Returning dummy frames."
                )
                return [
                    np.zeros((dummy_frame_height, dummy_frame_width, 3), dtype=np.uint8)
                    for _ in range(self.num_frames)
                ]
            return list(frames)
        except Exception as e:
            print(
                f"Error reading or sampling frames from {video_path}: {e}. Returning dummy frames."
            )
            return [
                np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.num_frames)
            ]

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]
        frames_list_hwc = self._sample_frames(video_path)

        processed_output = self.image_processor(frames_list_hwc, return_tensors="pt")
        pixel_values = processed_output.pixel_values

        if pixel_values.ndim == 5 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values.squeeze(0)

        if pixel_values.ndim != 4:
            error_message = (
                f"Critical Error: pixel_values for video {video_path} has {pixel_values.ndim} dimensions "
                f"(shape: {pixel_values.shape}) after processor and potential squeeze. Expected 4D. "
                f"Frames list length: {len(frames_list_hwc)}, first frame shape if available: {frames_list_hwc[0].shape if frames_list_hwc and hasattr(frames_list_hwc[0], 'shape') else 'N/A'}."
            )
            print(error_message)
            raise ValueError(error_message)

        if (
            pixel_values.shape[1] == self.num_frames
            and pixel_values.shape[0] != self.num_frames
        ):
            pixel_values = pixel_values.permute(1, 0, 2, 3)
        elif pixel_values.shape[0] == self.num_frames:
            pass
        else:
            print(
                f"Warning: pixel_values for video {video_path} has an unhandled 4D shape: {pixel_values.shape} "
                f"(num_frames expected: {self.num_frames}). "
                "The model expects input of shape (num_frames, num_channels, height, width) per item. "
                "Ensure video integrity and processor output alignment."
            )
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }

# --- Function to plot metrics ---
def plot_and_save_metrics(
    epochs_list, train_losses, val_losses, val_accuracies, train_lrs, filename
):
    fig, ax1 = plt.subplots(
        figsize=(12, 8)
    )  # Increased size for better readability with LR

    # Plotting training and validation loss
    color = "tab:red"
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(
        epochs_list,
        train_losses,
        color=color,
        linestyle="-",
        marker="o",
        label="Training Loss",
    )
    ax1.plot(
        epochs_list,
        val_losses,
        color=color,
        linestyle="--",
        marker="x",
        label="Validation Loss",
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.legend(loc="upper left")

    # Creating a second y-axis for validation accuracy
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Accuracy", color=color)
    ax2.plot(
        epochs_list,
        val_accuracies,
        color=color,
        linestyle=":",
        marker="s",
        label="Validation Accuracy",
    )
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.legend(loc="upper right")

    # Creating a third y-axis for learning rate
    ax3 = ax1.twinx()
    # Offset the third axis to prevent overlap with ax2's labels
    ax3.spines["right"].set_position(("outward", 60))  # Adjust 60 as needed
    color = "tab:green"
    ax3.set_ylabel("Learning Rate", color=color)
    # Plot average LR for each epoch, or LR at the end of epoch
    ax3.plot(
        epochs_list,
        train_lrs,
        color=color,
        linestyle="-.",
        marker="^",
        label="Learning Rate (End of Epoch)",
    )
    ax3.tick_params(axis="y", labelcolor=color)
    ax3.legend(loc="lower left")
    ax3.set_yscale("log")  # Often useful for LR plots

    fig.tight_layout()
    plt.title("Training and Validation Metrics")
    plt.savefig(filename)
    print(f"Metrics plot saved to {filename}")
    plt.close(fig)

# --- Main Training Script ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Load Image Processor and Model Configuration
    print(f"Loading image processor for {PRETRAINED_MODEL_NAME}...")
    image_processor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL_NAME)

    print(f"Loading model config for {PRETRAINED_MODEL_NAME}...")
    model_config = TimesformerConfig.from_pretrained(PRETRAINED_MODEL_NAME)
    NUM_FRAMES_TO_SAMPLE = model_config.num_frames
    print(f"Model expects {NUM_FRAMES_TO_SAMPLE} frames per video.")

    # 2. Create Datasets and DataLoaders
    print("Creating datasets...")
    train_dataset = VideoClassificationDataset(
        video_dir=TRAIN_DIR,
        image_processor=image_processor,
        num_frames=NUM_FRAMES_TO_SAMPLE,
    )
    val_dataset = VideoClassificationDataset(
        video_dir=VAL_DIR,
        image_processor=image_processor,
        num_frames=NUM_FRAMES_TO_SAMPLE,
        class_to_idx=train_dataset.class_to_idx,
    )

    actual_num_classes = len(train_dataset.class_names)
    print(f"Found {actual_num_classes} classes: {train_dataset.class_names}")
    if actual_num_classes != NUM_CLASSES:
        print(
            f"Warning: Configured NUM_CLASSES ({NUM_CLASSES}) differs from found classes ({actual_num_classes}). Using {actual_num_classes}."
        )
        # NUM_CLASSES = actual_num_classes # Use actual number of classes found

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    print("DataLoaders created.")

    # 3. Initialize Model and Optimizer
    print(f"Loading pre-trained model: {PRETRAINED_MODEL_NAME}...")
    model = TimesformerForVideoClassification.from_pretrained(
        PRETRAINED_MODEL_NAME,
        num_labels=actual_num_classes,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)
    print("Model loaded.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    accuracy_metric = evaluate.load("accuracy")

    # 4. Initialize Learning Rate Scheduler
    num_training_steps = NUM_EPOCHS * len(train_loader)
    num_warmup_steps = int(WARMUP_PROPORTION * num_training_steps)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    print(
        f"LR Scheduler: Cosine Annealing with Warmup. Total steps: {num_training_steps}, Warmup steps: {num_warmup_steps}"
    )

    # Lists to store metrics for plotting
    history_train_losses = []
    history_val_losses = []
    history_val_accuracies = []
    history_lrs = []  # To store learning rates
    epochs_ran = []

    # 5. Training Loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        epochs_ran.append(epoch + 1)
        # --- Training Phase ---
        model.train()
        train_loss_epoch = 0.0
        train_accuracy_calculator = evaluate.load("accuracy")
        progress_bar_train = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]"
        )

        current_epoch_lrs = []  # Store LRs within an epoch if needed, or just end of epoch LR
        for batch_idx, batch in enumerate(progress_bar_train):
            inputs = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(pixel_values=inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()
            lr_scheduler.step()  # Step the scheduler after optimizer update

            train_loss_epoch += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            train_accuracy_calculator.add_batch(
                predictions=predictions, references=labels
            )

            current_lr = optimizer.param_groups[0]["lr"]
            current_epoch_lrs.append(current_lr)

            if batch_idx % 20 == 0:  # Log less frequently
                progress_bar_train.set_postfix({"loss": loss.item(), "lr": current_lr})

        avg_train_loss = train_loss_epoch / len(train_loader)
        train_accuracy = train_accuracy_calculator.compute()["accuracy"]
        history_train_losses.append(avg_train_loss)
        history_lrs.append(
            optimizer.param_groups[0]["lr"]
        )  # LR at the end of the epoch
        print(
            f"Epoch {epoch + 1} [Train] - Avg Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # --- Validation Phase ---
        model.eval()
        val_loss_epoch = 0.0
        val_accuracy_calculator = evaluate.load("accuracy")
        progress_bar_val = tqdm(
            val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val]"
        )

        with torch.no_grad():
            for batch in progress_bar_val:
                inputs = batch["pixel_values"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                outputs = model(pixel_values=inputs, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                val_loss_epoch += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                val_accuracy_calculator.add_batch(
                    predictions=predictions, references=labels
                )
                progress_bar_val.set_postfix({"loss": loss.item()})

        avg_val_loss = val_loss_epoch / len(val_loader)
        val_accuracy = val_accuracy_calculator.compute()["accuracy"]
        history_val_losses.append(avg_val_loss)
        history_val_accuracies.append(val_accuracy)
        print(
            f"Epoch {epoch + 1} [Val]   - Avg Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}"
        )
        print("-" * 50)

    # Plot and save metrics after all epochs are done
    if NUM_EPOCHS > 0:
        plot_and_save_metrics(
            epochs_ran,
            history_train_losses,
            history_val_losses,
            history_val_accuracies,
            history_lrs,  # Pass recorded LRs
            PLOT_FILENAME,
        )

    print("Training finished.")

    # Optional: Save the fine-tuned model
    output_dir = "./timesformer_finetuned_custom"
    model.save_pretrained(output_dir)
    image_processor.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
