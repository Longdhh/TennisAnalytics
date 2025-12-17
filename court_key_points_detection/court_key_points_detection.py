# from torchvision.models import resnet18
from torchvision.models import resnet50
from torchvision import models
from torchvision import transforms
import torch
import constants
import cv2

class CourtKeyPointsDetection:
    def __init__(self):
        # self.model = resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model = resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        self.model.load_state_dict(torch.load(constants.TENNIS_COURT_DETECTION_MODEL_PATH))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(tensor)

        key_points = outputs.squeeze().numpy()
        h, w = image.shape[:2]
        key_points[::2] *= w / 224.0
        key_points[1::2] *= h / 224.0

        return key_points

