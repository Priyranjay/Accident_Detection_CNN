import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import gradio as gr
import cv2
from PIL import Image
import os
import tempfile
from twilio.rest import Client  # Import Twilio client

#Twilio Configurationpp
TWILIO_ACCOUNT_SID = "ACf350659118d4ac7ec4ad6ac29be52bac"
TWILIO_AUTH_TOKEN = "6b86ed8730f48feec657b42d6d91917e"
TWILIO_PHONE_NUMBER = "++16592710525"      
TARGET_PHONE_NUMBER = "+917903946154"        

def send_sms_alert(message_body):
    """Helper function to authenticate and send an SMS alert via Twilio."""
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=TARGET_PHONE_NUMBER
        )
        print(f"SMS Alert sent successfully! SID: {message.sid}")
    except Exception as e:
        print(f"Failed to send Twilio SMS: {e}")


# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# 2. DEFINE YOUR TEXT LABELS
LABELS = ["Accident", "Non-Accident"]

# 3. INITIALIZE AND LOAD WEIGHTS
model = SimpleCNN()
model.load_state_dict(torch.load(r"C:\Users\Priyranjay Gupta\Downloads\CNN_model.pth", map_location=torch.device('cpu')))
model.eval() 

# 4. DEFINE IMAGE PREPROCESSING
preprocess = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. VIDEO PREPROCESSING AND DETECTION FUNCTION
def predict_video(video_path):
    if video_path is None:
        return None, "Please upload a video file."
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error opening video file."

    frame_count = 0
    best_accident_prob = 0.0
    best_frame_number = None
    best_frame_path = None
    alert_status = "No SMS sent."
    accident_threshold = 0.75

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_t = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_t)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze(0)

        accident_prob = float(probabilities[0])
        if accident_prob > best_accident_prob:
            best_accident_prob = accident_prob
            best_frame_number = frame_count
            if best_accident_prob >= accident_threshold:
                if best_frame_path is not None:
                    try:
                        os.remove(best_frame_path)
                    except Exception:
                        pass
                best_frame_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
                cv2.imwrite(best_frame_path, frame)

        # Overlay the label and probability on the frame
        label_text = f"Accident: {accident_prob:.2%}" if accident_prob > 0.5 else f"No Accident: {1-accident_prob:.2%}"
        color = (0, 0, 255) if accident_prob > 0.5 else (0, 255, 0)
        cv2.putText(frame, label_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    cap.release()

    if best_accident_prob >= accident_threshold:
        alert_text = f"High-confidence accident detected at frame {best_frame_number} ({best_accident_prob:.2%})."
        send_sms_alert(alert_text)
        alert_status = "SMS alert sent."
    else:
        alert_text = "No high-confidence accident frame detected."
        best_frame_path = None

    summary_text = f"{alert_text} Processed {frame_count} frames."
    if best_frame_number is not None:
        summary_text += f" Highest accident probability was {best_accident_prob:.2%} at frame {best_frame_number}."

    return best_frame_path, summary_text

# 6. LAUNCH GRADIO VIDEO WEB UI
interface = gr.Interface(
    fn=predict_video,
    inputs=gr.Video(label="Upload video"), 
    outputs=[
        gr.Image(type="filepath", label="High-Confidence Accident Frame"),
        gr.Textbox(label="Detection Summary")
    ],
    title="SimpleCNN Accident Frame Detector with Twilio SMS Alerts",
    description="Upload a video file. The tool analyzes each frame and returns up to 4 detected accident frames plus a summary."
)

if __name__ == "__main__":
    interface.launch()
