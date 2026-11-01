
import os
import json
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import vit_b_16

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "vit_sipakmed.pth")
CONFIG_PATH = os.path.join(BASE_DIR, "sipakmed_classes.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CLASSES = json.load(f)

ABNORMAL_IDX = [
    CLASSES.index("im_Dyskeratotic"),
    CLASSES.index("im_Koilocytotic"),
]

FLAG_THRESHOLD = 0.40
HIGH_RISK_MIN = 0.70

DISCLAIMER = (
    "Educational demonstration only. This AI output is not a diagnosis "
    "and must not replace professional medical evaluation."
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

model = None


def get_model():
    global model

    if model is None:
        loaded_model = vit_b_16(weights=None)
        loaded_model.heads.head = nn.Linear(
            loaded_model.heads.head.in_features,
            len(CLASSES)
        )

        state_dict = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )

        loaded_model.load_state_dict(state_dict)
        model = loaded_model.to(DEVICE).eval()

    return model


def risk_level(score):
    if score >= HIGH_RISK_MIN:
        return "High"
    if score >= FLAG_THRESHOLD:
        return "Moderate"
    return "Low"


RECOMMENDATION = {
    "Low": "No strong abnormal features detected. Continue routine screening "
           "as advised by your doctor.",
    "Moderate": "Some abnormal features detected. Please arrange a follow-up "
                "with a healthcare professional.",
    "High": "Significant abnormal features detected. Please consult a "
            "healthcare professional promptly for further evaluation.",
}


def predict_risk(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    current_model = get_model()

    with torch.no_grad():
        probs = torch.softmax(current_model(tensor), dim=1)[0].cpu()

    score = float(probs[ABNORMAL_IDX].sum())
    level = risk_level(score)

    return {
        "predicted_cell_type": CLASSES[int(probs.argmax())],
        "class_probabilities": {
            c: round(float(p), 4)
            for c, p in zip(CLASSES, probs)
        },
        "abnormality_score": round(score, 4),
        "risk_indicator": level,
        "recommendation": RECOMMENDATION[level],
        "disclaimer": DISCLAIMER,
    }
