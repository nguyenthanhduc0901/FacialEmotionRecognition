import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import time

# --- Configuration ---
# Điều chỉnh các giá trị này cho phù hợp với cấu hình huấn luyện của bạn
CONFIG = {
    "model_name": "resnet18",  # Phải khớp với model_name đã dùng để huấn luyện
    "num_classes": 7,
    "image_size": 128,  # Phải khớp với image_size đã dùng để huấn luyện
    "model_path": "best_resnet18_emotion_model.pth",  # SỬA ĐƯỜNG DẪN NÀY
    "haar_cascade_path": cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
    "camera_index": 0,  # 0 thường là webcam tích hợp, thử 1, 2,... nếu không được
}

# Ánh xạ nhãn cảm xúc (phải khớp với cách bạn đã huấn luyện)
EMOTION_LABELS = {
    0: "Happy",
    1: "Surprise",
    2: "Sad",
    3: "Angry",
    4: "Disgust",
    5: "Fear",
    6: "Neutral",
}
# --- End Configuration ---


# Hàm get_pretrained_model (sao chép từ script huấn luyện của bạn)
def get_pretrained_model(model_name="resnet18", num_classes=7, pretrained=True):
    model = None
    # Khi tải model để inference, pretrained=False là đủ nếu bạn tải weights của mình
    # Tuy nhiên, để cấu trúc model giống hệt, chúng ta có thể giữ logic này
    # và sau đó ghi đè weights bằng model.load_state_dict()
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Tải model đã huấn luyện
    model = get_pretrained_model(
        model_name=CONFIG["model_name"],
        num_classes=CONFIG["num_classes"],
        pretrained=False,  # False vì chúng ta sẽ tải weights đã huấn luyện của mình
    )
    try:
        model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file model tại '{CONFIG['model_path']}'")
        print("Vui lòng kiểm tra lại đường dẫn trong CONFIG['model_path'].")
        return
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        print("Đảm bảo model_name và num_classes trong CONFIG khớp với model đã lưu.")
        return

    model.to(device)
    model.eval()  # Quan trọng: đặt model ở chế độ đánh giá

    # 2. Định nghĩa phép biến đổi ảnh (giống như val_test_transform)
    transform = transforms.Compose(
        [
            transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 3. Tải bộ phát hiện khuôn mặt Haar Cascade
    if not os.path.exists(CONFIG["haar_cascade_path"]):
        print(
            f"Lỗi: Không tìm thấy file Haar Cascade tại '{CONFIG['haar_cascade_path']}'"
        )
        print(
            "Bạn có thể cần cài đặt đầy đủ opencv-python hoặc cung cấp đường dẫn chính xác."
        )
        return
    face_cascade = cv2.CascadeClassifier(CONFIG["haar_cascade_path"])

    # 4. Mở webcam
    cap = cv2.VideoCapture(CONFIG["camera_index"])
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở webcam (index: {CONFIG['camera_index']}).")
        print("Hãy thử các index khác như 1, 2, ... hoặc kiểm tra driver webcam.")
        return

    print("Nhấn 'q' để thoát.")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi: Không thể đọc khung hình từ webcam.")
            break

        # Lật khung hình theo chiều ngang để có hiệu ứng gương (tùy chọn)
        frame = cv2.flip(frame, 1)

        # Chuyển sang ảnh xám để phát hiện khuôn mặt
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Phát hiện khuôn mặt
        # Bạn có thể điều chỉnh các tham số scaleFactor và minNeighbors
        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        for x, y, w, h in faces:
            # Cắt vùng khuôn mặt từ khung hình màu (BGR)
            face_roi_bgr = frame[y : y + h, x : x + w]

            # Chuyển BGR (OpenCV) sang RGB (PIL)
            face_roi_rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_roi_rgb)

            # Áp dụng phép biến đổi và chuẩn bị tensor đầu vào
            input_tensor = transform(pil_image)
            input_batch = input_tensor.unsqueeze(0).to(device)  # Thêm batch dimension

            # Thực hiện dự đoán
            with torch.no_grad():
                outputs = model(input_batch)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

            predicted_emotion = EMOTION_LABELS.get(predicted_idx.item(), "Unknown")
            confidence_score = confidence.item()

            # Vẽ hình chữ nhật quanh khuôn mặt và hiển thị cảm xúc
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label_text = f"{predicted_emotion} ({confidence_score:.2f})"

            # Đặt text phía trên hình chữ nhật
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1
            )
            text_y_pos = (
                y - 10 if y - 10 > 10 else y + h + 20
            )  # Đảm bảo text không ra ngoài khung hình

            # Vẽ một background nhỏ cho text để dễ đọc hơn
            cv2.rectangle(
                frame,
                (x, text_y_pos - text_height - baseline),
                (x + text_width, text_y_pos + baseline),
                (0, 0, 0),
                -1,
            )  # Nền đen
            cv2.putText(
                frame,
                label_text,
                (x, text_y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Tính toán và hiển thị FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:  # Cập nhật FPS mỗi giây
            fps = frame_count / elapsed_time
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )
            frame_count = 0
            start_time = time.time()

        # Hiển thị khung hình kết quả
        cv2.imshow("Webcam Emotion Recognition", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
    print("Đã đóng webcam và cửa sổ.")


if __name__ == "__main__":
    main()
