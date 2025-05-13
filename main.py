import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm.notebook import (
    tqdm,
)  # Sử dụng tqdm.notebook cho Jupyter, hoặc from tqdm import tqdm cho script thường
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # Nếu bạn vẫn muốn dùng TensorBoard
from PIL import Image

# from torch.cuda.amp import GradScaler, autocast # Cũ
from torch.amp import autocast  # Mới, cho device 'cuda'
from torch.cuda.amp import GradScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import copy

# Thiết lập seed để có thể tái tạo kết quả
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Tắt benchmark để tăng tính tái tạo

# --- Configuration ---
CONFIG = {
    "model_name": "resnet18",
    "num_classes": 7,
    "batch_size": 64,
    "image_size": 128,
    "num_epochs": 70,  # Sẽ được điều chỉnh bởi Early Stopping
    "max_lr": 1e-3,  # Learning rate tối đa cho OneCycleLR
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,  # Giữ lại label smoothing
    "grad_clip_value": 1.0,
    "use_amp": torch.cuda.is_available(),
    "tensorboard_log_dir": "/kaggle/working/runs/emotion_onecyclelr",  # Thay đổi tên log
    "data_base_path": "/kaggle/input/emotions/data/",  # Đảm bảo đúng
    "num_workers_cap": min(4, os.cpu_count() if os.cpu_count() else 1),
    "patience_early_stopping": 10,  # Số epoch chờ trước khi dừng sớm
    "output_dir": "/kaggle/working/",
}

# Ánh xạ nhãn (QUAN TRỌNG: phải khớp với dữ liệu và thứ tự mong muốn cho model)
# Ví dụ, nếu file CSV của bạn có cột nhãn là chuỗi và EMOTION_LABELS_MAP ánh xạ
# các chuỗi đó sang số (0-6) theo thứ tự bạn muốn.
# Đây là ví dụ dựa trên phân phối bạn cung cấp, bạn cần kiểm tra và sửa lại cho đúng
EMOTION_LABELS_MAP = {
    0: "Happy",  # Lớp 0
    1: "Surprise",  # Lớp 1
    2: "Sad",  # Lớp 2
    3: "Angry",  # Lớp 3
    4: "Disgust",  # Lớp 4
    5: "Fear",  # Lớp 5
    6: "Neutral",  # Lớp 6
}
# --- End Configuration ---


class Four4All(Dataset):
    def __init__(
        self, csv_file, img_dir, transform=None, image_size=(64, 64), label_map=None
    ):
        # Đọc file CSV, giả sử cột đầu là image_id, cột hai là label_str (tên cảm xúc)
        try:
            self.raw_labels_df = pd.read_csv(
                csv_file, header=None, names=["image_id", "label_str"]
            )
        except pd.errors.EmptyDataError:
            print(f"Lỗi: File CSV {csv_file} rỗng hoặc không thể đọc.")
            self.labels_df = pd.DataFrame(
                columns=["image_id", "label"]
            )  # Trả về dataframe rỗng
            return
        except FileNotFoundError:
            print(f"Lỗi: File CSV {csv_file} không tìm thấy.")
            self.labels_df = pd.DataFrame(columns=["image_id", "label"])
            return

        self.img_dir = img_dir
        self.transform = transform
        self.image_size = image_size
        self.label_map = label_map

        if self.label_map:
            name_to_int_map = {v: k for k, v in self.label_map.items()}
            if pd.api.types.is_string_dtype(self.raw_labels_df["label_str"]):
                self.raw_labels_df["label"] = self.raw_labels_df["label_str"].map(
                    name_to_int_map
                )
                if self.raw_labels_df["label"].isnull().any():
                    unknown_labels = self.raw_labels_df[
                        self.raw_labels_df["label"].isnull()
                    ]["label_str"].unique()
                    print(
                        f"Cảnh báo: Tìm thấy {len(unknown_labels)} nhãn không xác định trong {csv_file}: {unknown_labels}. Các mẫu này sẽ bị bỏ qua."
                    )
                    self.raw_labels_df = self.raw_labels_df.dropna(subset=["label"])
                if not self.raw_labels_df.empty:
                    self.raw_labels_df["label"] = self.raw_labels_df["label"].astype(
                        int
                    )
                else:
                    print(
                        f"Cảnh báo: Không còn mẫu nào trong {csv_file} sau khi loại bỏ nhãn không xác định."
                    )

            elif pd.api.types.is_numeric_dtype(self.raw_labels_df["label_str"]):
                self.raw_labels_df["label"] = self.raw_labels_df["label_str"].astype(
                    int
                )
                # Kiểm tra xem các nhãn số có nằm trong phạm vi của label_map không
                valid_numeric_labels = self.raw_labels_df["label"].isin(
                    self.label_map.keys()
                )
                if not valid_numeric_labels.all():
                    invalid_numeric_count = (~valid_numeric_labels).sum()
                    print(
                        f"Cảnh báo: {invalid_numeric_count} nhãn số trong {csv_file} nằm ngoài phạm vi của EMOTION_LABELS_MAP. Các mẫu này sẽ bị bỏ qua."
                    )
                    self.raw_labels_df = self.raw_labels_df[valid_numeric_labels]
            else:
                raise ValueError(
                    f"Định dạng cột nhãn không được hỗ trợ trong {csv_file}."
                )
        else:  # Nếu không có label_map, giả sử nhãn đã là số và đúng
            if pd.api.types.is_numeric_dtype(self.raw_labels_df["label_str"]):
                self.raw_labels_df["label"] = self.raw_labels_df["label_str"].astype(
                    int
                )
            else:  # Nếu là chuỗi mà không có map thì lỗi
                raise ValueError(
                    f"Cột nhãn trong {csv_file} là chuỗi nhưng không có label_map được cung cấp."
                )

        if not self.raw_labels_df.empty and "label" in self.raw_labels_df.columns:
            self.labels_df = self.raw_labels_df[["image_id", "label"]].copy()
            self._validate_files()
        else:
            self.labels_df = pd.DataFrame(columns=["image_id", "label"])  # Dataset rỗng
            print(f"Dataset {csv_file} rỗng sau khi xử lý nhãn.")

    def _validate_files(self):
        if self.labels_df.empty:
            print(
                f"Không có mẫu nào để validate trong {self.img_dir} do labels_df rỗng."
            )
            return

        # print(f"Validating image files in {self.img_dir}...") # Bỏ bớt print
        valid_indices = []
        missing_count = 0
        for idx in range(len(self.labels_df)):
            img_name = os.path.join(
                self.img_dir, str(self.labels_df.iloc[idx, 0])
            )  # Đảm bảo tên file là chuỗi
            if os.path.exists(img_name):
                valid_indices.append(idx)
            else:
                missing_count += 1

        if missing_count > 0:
            print(
                f"Warning: {missing_count} image files were not found in {self.img_dir} and associated samples will be skipped."
            )

        self.labels_df = self.labels_df.iloc[valid_indices].reset_index(drop=True)
        # print(f"Dataset initialized with {len(self.labels_df)} valid samples from {self.img_dir}.") # Bỏ bớt print

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, str(self.labels_df.iloc[idx, 0]))
        label = self.labels_df.iloc[idx, 1]

        try:
            image = Image.open(img_name).convert("RGB")
        except FileNotFoundError:
            # print(f"ERROR during getitem: Image not found at {img_name}. Returning placeholder.")
            image = Image.new("RGB", self.image_size, color="black")
        except Exception as e:
            # print(f"ERROR loading image {img_name}: {e}. Returning placeholder.")
            image = Image.new("RGB", self.image_size, color="black")

        if self.transform:
            image = self.transform(image)
        return image, label


def get_pretrained_model(model_name="resnet18", num_classes=7, pretrained=True):
    model = None
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported or implemented yet.")
    return model


def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["tensorboard_log_dir"], exist_ok=True)

    writer = SummaryWriter(CONFIG["tensorboard_log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device. AMP enabled: {CONFIG['use_amp']}")
    print(f"Output directory: {CONFIG['output_dir']}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),  # Giảm nhẹ rotation
            transforms.RandomAffine(
                degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5
            ),  # Giảm nhẹ Affine
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
            ),  # Giảm nhẹ ColorJitter
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(
                p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0, inplace=False
            ),  # Giảm nhẹ RandomErasing
        ]
    )

    val_test_transform = transforms.Compose(
        [
            transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    num_workers = min(
        CONFIG["num_workers_cap"], os.cpu_count() if os.cpu_count() else 1
    )
    print(f"Using {num_workers} workers for DataLoaders.")
    current_image_size_tuple = (CONFIG["image_size"], CONFIG["image_size"])

    train_dataset = Four4All(
        csv_file=os.path.join(CONFIG["data_base_path"], "train_labels.csv"),
        img_dir=os.path.join(CONFIG["data_base_path"], "train"),
        transform=train_transform,
        image_size=current_image_size_tuple,
        label_map=EMOTION_LABELS_MAP,
    )
    if len(train_dataset) == 0:
        print("Lỗi: Train dataset rỗng. Kiểm tra lại đường dẫn và file CSV.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Train loader: {len(train_loader)} batches, {len(train_dataset)} samples")

    val_dataset = Four4All(
        csv_file=os.path.join(CONFIG["data_base_path"], "val_labels.csv"),
        img_dir=os.path.join(CONFIG["data_base_path"], "val"),
        transform=val_test_transform,
        image_size=current_image_size_tuple,
        label_map=EMOTION_LABELS_MAP,
    )
    if len(val_dataset) == 0:
        print("Lỗi: Validation dataset rỗng. Kiểm tra lại đường dẫn và file CSV.")
        return
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"Validation loader: {len(val_loader)} batches, {len(val_dataset)} samples")

    model = get_pretrained_model(
        model_name=CONFIG["model_name"],
        num_classes=CONFIG["num_classes"],
        pretrained=True,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters in {CONFIG['model_name']}: {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    print(
        "Criterion: CrossEntropyLoss with label smoothing (NO class weights this time)."
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["max_lr"],  # OneCycleLR sẽ tự điều chỉnh LR, đây là LR max
        weight_decay=CONFIG["weight_decay"],
    )

    # Sử dụng OneCycleLR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG["max_lr"],
        epochs=CONFIG["num_epochs"],
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # Tỷ lệ số bước tăng LR
        div_factor=25,  # max_lr / initial_lr
        final_div_factor=1e4,  # max_lr / min_lr
    )
    print("Using OneCycleLR scheduler.")

    scaler = GradScaler(enabled=CONFIG["use_amp"])
    best_val_metric = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    early_stop_triggered = False

    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        running_loss_train = 0.0
        correct_train = 0
        total_train = 0
        progress_bar_train = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']} [Train]",
            unit="batch",
            leave=False,
        )

        for inputs, labels in progress_bar_train:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)

            # Cập nhật autocast
            with autocast(
                device_type="cuda", dtype=torch.float16, enabled=CONFIG["use_amp"]
            ):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            if CONFIG["grad_clip_value"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), CONFIG["grad_clip_value"]
                )
            scaler.step(optimizer)
            scaler.update()

            # OneCycleLR được step sau mỗi batch
            if scheduler is not None and isinstance(
                scheduler, optim.lr_scheduler.OneCycleLR
            ):
                scheduler.step()

            running_loss_train += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            current_lr = optimizer.param_groups[0]["lr"]
            progress_bar_train.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=(
                    f"{100.0 * correct_train / total_train:.2f}%"
                    if total_train > 0
                    else "0.00%"
                ),
                lr=f"{current_lr:.2e}",
            )

        train_epoch_loss = running_loss_train / total_train if total_train > 0 else 0
        train_epoch_acc = correct_train / total_train if total_train > 0 else 0
        writer.add_scalar("Loss/train", train_epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", train_epoch_acc, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)

        model.eval()
        running_loss_val = 0.0
        correct_val = 0
        total_val = 0
        all_val_preds = []
        all_val_labels = []
        progress_bar_val = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']} [Valid]",
            unit="batch",
            leave=False,
        )

        with torch.no_grad():
            for inputs, labels in progress_bar_val:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                    device, non_blocking=True
                )
                with autocast(
                    device_type="cuda", dtype=torch.float16, enabled=CONFIG["use_amp"]
                ):
                    outputs = model(inputs)
                    loss_val = criterion(outputs, labels)

                running_loss_val += loss_val.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                progress_bar_val.set_postfix(
                    loss=f"{loss_val.item():.4f}",
                    acc=(
                        f"{100.0 * correct_val / total_val:.2f}%"
                        if total_val > 0
                        else "0.00%"
                    ),
                )

        val_epoch_loss = running_loss_val / total_val if total_val > 0 else 0
        val_epoch_acc = correct_val / total_val if total_val > 0 else 0
        writer.add_scalar("Loss/validation", val_epoch_loss, epoch)
        writer.add_scalar("Accuracy/validation", val_epoch_acc, epoch)

        val_f1_macro = 0.0
        if total_val > 0:
            # Đảm bảo target_names cho report_val khớp với số lớp thực tế
            unique_val_labels = np.unique(all_val_labels)
            report_val_target_names = [
                EMOTION_LABELS_MAP[i]
                for i in sorted(EMOTION_LABELS_MAP.keys())
                if i in unique_val_labels or i in np.unique(all_val_preds)
            ]

            # Lấy danh sách các nhãn số thực sự có trong validation hoặc prediction để report
            actual_labels_for_report = sorted(
                list(set(all_val_labels) | set(all_val_preds))
            )

            if actual_labels_for_report:  # Chỉ tính report nếu có nhãn
                report_val = classification_report(
                    all_val_labels,
                    all_val_preds,
                    output_dict=True,
                    zero_division=0,
                    labels=actual_labels_for_report,
                )
                val_f1_macro = report_val.get("macro avg", {}).get(
                    "f1-score", 0.0
                )  # An toàn hơn
            writer.add_scalar("F1_Macro/validation", val_f1_macro, epoch)

        print(
            f"Epoch {epoch+1}/{CONFIG['num_epochs']} -> "
            f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc:.4f} | "
            f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}, Val F1 Macro: {val_f1_macro:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        current_val_metric = val_f1_macro
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(
                best_model_wts,
                os.path.join(
                    CONFIG["output_dir"],
                    f"best_{CONFIG['model_name']}_emotion_model.pth",
                ),
            )
            print(
                f"🎉 New best model saved! Val F1 Macro: {best_val_metric:.4f} (Acc: {val_epoch_acc:.4f})"
            )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= CONFIG["patience_early_stopping"]:
            print(
                f"🛑 Early stopping triggered after {CONFIG['patience_early_stopping']} epochs with no improvement."
            )
            early_stop_triggered = True
            break

    if not early_stop_triggered:
        print("Training finished (completed all epochs).")
    torch.save(
        model.state_dict(),
        os.path.join(
            CONFIG["output_dir"], f"final_{CONFIG['model_name']}_emotion_model.pth"
        ),
    )
    print(f"Final model saved as final_{CONFIG['model_name']}_emotion_model.pth")

    if os.path.exists(
        os.path.join(
            CONFIG["output_dir"], f"best_{CONFIG['model_name']}_emotion_model.pth"
        )
    ):
        print(
            f"Loading best model for final evaluation (Best Val F1 Macro: {best_val_metric:.4f})..."
        )
        model.load_state_dict(
            torch.load(
                os.path.join(
                    CONFIG["output_dir"],
                    f"best_{CONFIG['model_name']}_emotion_model.pth",
                )
            )
        )
    else:
        print("No best model found, using final model for evaluation.")

    # --- Evaluation on Test Set ---
    test_dataset = Four4All(
        csv_file=os.path.join(CONFIG["data_base_path"], "test_labels.csv"),
        img_dir=os.path.join(CONFIG["data_base_path"], "test"),
        transform=val_test_transform,
        image_size=current_image_size_tuple,
        label_map=EMOTION_LABELS_MAP,
    )
    if len(test_dataset) == 0:
        print("Cảnh báo: Test dataset rỗng. Bỏ qua đánh giá trên tập test.")
    else:
        test_loader = DataLoader(
            test_dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        print(f"Test loader: {len(test_loader)} batches, {len(test_dataset)} samples")

        print(f"\n--- Evaluating Best Model ({CONFIG['model_name']}) on Test Set ---")
        model.eval()
        test_correct = 0
        test_total = 0
        test_running_loss = 0.0
        all_test_preds = []
        all_test_labels = []
        test_criterion = nn.CrossEntropyLoss(
            label_smoothing=CONFIG["label_smoothing"]
        )  # Không có weight

        with torch.no_grad():
            progress_bar_test = tqdm(
                test_loader, desc=f"Testing Best {CONFIG['model_name']}", unit="batch"
            )
            for inputs, labels_batch in progress_bar_test:
                inputs, labels_batch_dev = inputs.to(
                    device, non_blocking=True
                ), labels_batch.to(device, non_blocking=True)
                with autocast(
                    device_type="cuda", dtype=torch.float16, enabled=CONFIG["use_amp"]
                ):
                    outputs = model(inputs)
                    loss = test_criterion(outputs, labels_batch_dev)

                test_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels_batch_dev.size(0)
                test_correct += (predicted == labels_batch_dev).sum().item()
                all_test_preds.extend(predicted.cpu().numpy())
                all_test_labels.extend(labels_batch.cpu().numpy())
                progress_bar_test.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=(
                        f"{100.0 * test_correct / test_total:.2f}%"
                        if test_total > 0
                        else "0.00%"
                    ),
                )

        if test_total > 0:
            final_test_loss = test_running_loss / test_total
            final_test_acc = test_correct / test_total
            print(f"\nBest Model ({CONFIG['model_name']}) Test Results:")
            print(f"  Loss: {final_test_loss:.4f}")
            print(f"  Accuracy: {final_test_acc:.4f} ({test_correct}/{test_total})")

            if all_test_labels and all_test_preds:
                target_names_report = [
                    EMOTION_LABELS_MAP[i] for i in sorted(EMOTION_LABELS_MAP.keys())
                ]
                actual_test_labels_for_report = sorted(
                    list(set(all_test_labels) | set(all_test_preds))
                )

                if actual_test_labels_for_report:
                    print("\nClassification Report (Test Set):")
                    report_str = classification_report(
                        all_test_labels,
                        all_test_preds,
                        target_names=[
                            EMOTION_LABELS_MAP[i] for i in actual_test_labels_for_report
                        ],  # Chỉ tên cho các lớp có mặt
                        digits=4,
                        zero_division=0,
                        labels=actual_test_labels_for_report,
                    )
                    print(report_str)

                    cm = confusion_matrix(
                        all_test_labels,
                        all_test_preds,
                        labels=actual_test_labels_for_report,
                    )
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=[
                            EMOTION_LABELS_MAP[i] for i in actual_test_labels_for_report
                        ],
                        yticklabels=[
                            EMOTION_LABELS_MAP[i] for i in actual_test_labels_for_report
                        ],
                    )
                    plt.xlabel("Predicted Label", fontsize=12)
                    plt.ylabel("True Label", fontsize=12)
                    plt.title(
                        f"Confusion Matrix - Test Set - Best {CONFIG['model_name']}",
                        fontsize=15,
                    )
                    plt.tight_layout()
                    cm_fig_path = os.path.join(
                        CONFIG["output_dir"],
                        f"confusion_matrix_test_best_{CONFIG['model_name']}_onecycle.png",
                    )
                    plt.savefig(cm_fig_path)
                    print(f"\nConfusion matrix for test set saved to {cm_fig_path}")
                    plt.show()
        else:
            print(
                "Test set was empty or no predictions made. Skipping test evaluation metrics."
            )

    writer.close()


if __name__ == "__main__":
    # Kiểm tra xem có đang chạy trong Jupyter không để dùng tqdm phù hợp
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            from tqdm.notebook import tqdm

            print("Running in Jupyter Notebook, using tqdm.notebook.")
        else:
            from tqdm import tqdm

            print("Not in Jupyter Notebook, using standard tqdm.")
    except NameError:
        from tqdm import tqdm  # Mặc định nếu get_ipython không được định nghĩa

        print("Standard tqdm imported (likely not in Jupyter).")
    main()
