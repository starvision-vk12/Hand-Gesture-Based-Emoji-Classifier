import cv2
import torch
import torch.nn as nn
import numpy as np
import os
from torchvision import transforms

class CNN(nn.Module):
    def __init__(self, num_classes=12):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=5, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 5 * 5, 1024)
        self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.sigmoid(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN(num_classes=12)
model.load_state_dict(torch.load('emojinator.pth'))
model.eval()

def main():
    emojis = get_emojis()
    if not emojis:
        print("Error: No emojis loaded.")
        return

    cap = cv2.VideoCapture(0)
    x, y, w, h = 300, 50, 350, 350

    while cap.isOpened():
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(hsv, np.array([2, 50, 60]), np.array([25, 150, 255]))
        res = cv2.bitwise_and(img, img, mask=mask2)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        median = cv2.GaussianBlur(gray, (5, 5), 0)

        kernel_square = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(median, kernel_square, iterations=2)
        opening = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_square)
        ret, thresh = cv2.threshold(opening, 30, 255, cv2.THRESH_BINARY)

        thresh = thresh[y:y + h, x:x + w]
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 2500:
                x, y, w1, h1 = cv2.boundingRect(contour)
                newImage = thresh[y:y + h1, x:x + w1]
                newImage = cv2.resize(newImage, (50, 50))
                pred_probab, pred_class = pytorch_predict(model, newImage)
                print(pred_class, pred_probab)
                
                if pred_class < len(emojis):
                    img = overlay(img, emojis[pred_class], 400, 250, 90, 90)
                else:
                    print(f"Warning: Predicted class {pred_class} exceeds emoji list length.")

        x, y, w, h = 300, 50, 350, 350
        cv2.imshow("Frame", img)
        cv2.imshow("Contours", thresh)
        k = cv2.waitKey(10)
        if k == 27:
            break

def pytorch_predict(model, image):
    processed = pytorch_process_image(image)
    with torch.no_grad():
        output = model(processed)
        pred_probab = torch.softmax(output, dim=1).cpu().numpy()
        pred_class = np.argmax(pred_probab)
    return np.max(pred_probab), pred_class

def pytorch_process_image(img):
    image_x = 50
    image_y = 50
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_x, image_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img = transform(img)
    img = img.unsqueeze(0)  
    return img

def get_emojis():
    emojis_folder = 'hand_emo/'
    emojis = []
    for emoji in range(len(os.listdir(emojis_folder))):
        file_path = os.path.join(emojis_folder, f"{emoji}.png")
        emoji_image = cv2.imread(file_path, -1)
        if emoji_image is not None:
            emojis.append(emoji_image)
        else:
            print(f"Warning: Unable to load {file_path}")
    return emojis

def overlay(image, emoji, x, y, w, h):
    emoji = cv2.resize(emoji, (w, h))
    try:
        image[y:y+h, x:x+w] = blend_transparent(image[y:y+h, x:x+w], emoji)
    except:
        pass
    return image

def blend_transparent(face_img, overlay_t_img):
    overlay_img = overlay_t_img[:,:,:3]  
    overlay_mask = overlay_t_img[:,:,3:]  

    background_mask = 255 - overlay_mask

    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

pytorch_predict(model, np.zeros((50, 50, 1), dtype=np.uint8))
main()
