import cv2
import cv2.data
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from approach.ResEmoteNet import ResEmoteNet

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Emotions labels
emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']

model = ResEmoteNet().to(device)
checkpoint = torch.load('model/rafdb_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Settings for text
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
font_color = (0, 255, 0)  # This is BGR color
thickness = 3
line_type = cv2.LINE_AA

max_emotion = ''

def detect_emotion(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
    scores = probabilities.cpu().numpy().flatten()
    rounded_scores = [round(score, 2) for score in scores]
    return rounded_scores

def get_max_emotion(x, y, w, h, image):
    crop_img = image[y : y + h, x : x + w]
    pil_crop_img = Image.fromarray(crop_img)
    rounded_scores = detect_emotion(pil_crop_img)    
    max_index = np.argmax(rounded_scores)
    max_emotion = emotions[max_index]
    return max_emotion

def print_max_emotion(x, y, image, max_emotion):
    org = (x, y - 15)
    cv2.putText(image, max_emotion, org, font, font_scale, font_color, thickness, line_type)
    
def print_all_emotion(x, y, w, h, image):
    crop_img = image[y : y + h, x : x + w]
    pil_crop_img = Image.fromarray(crop_img)
    rounded_scores = detect_emotion(pil_crop_img)
    start_x = x + w + 10
    start_y = y - 20
    line_spacing = 100

    for index, value in enumerate(emotions):
        emotion_str = (f'{value}: {rounded_scores[index]:.2f}')
        position = (start_x, start_y + index * line_spacing)
        cv2.putText(image, emotion_str, position, font, font_scale, font_color, thickness, line_type)
    
def detect_bounding_box(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    print(faces)
    for (x, y, w, h) in faces:
        # Draw bounding box on face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        max_emotion = get_max_emotion(x, y, w, h, image)
        print_max_emotion(x, y, image, max_emotion)
        print_all_emotion(x, y, w, h, image)
    
    return faces

# Load the image file
image = cv2.imread('test_image/bright_angry.jpg')

# Process the image
faces = detect_bounding_box(image)


# Display the processed image
cv2.namedWindow("ResEmoteNet Image Test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ResEmoteNet Image Test", 800, 600)
cv2.imshow("ResEmoteNet Image Test", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
