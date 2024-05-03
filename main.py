import sys
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as func
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QTextEdit, QFileDialog
from PyQt5.QtGui import QPixmap

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the trained model
model_path = "model.pt"  # Replace with the path to your trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define the image preprocessing transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function for image classification using the trained model
def classify_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Failed to load the image."
    
    # Convert the NumPy array to PIL image
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Apply transformations
    image_tensor = transform(image_pil).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        if predicted.item() == 1:
            return "Cat"
        else:
            return "Dog"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Cat vs Dog Classifier")
        self.setGeometry(100, 100, 600, 400)

        # Button to load image
        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image)

        # Text box to display classification result
        self.classification_output = QTextEdit()
        self.classification_output.setReadOnly(True)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.load_image_button)
        main_layout.addWidget(QLabel("Classification Result:"))
        main_layout.addWidget(self.classification_output)
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

    def load_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                image_path = selected_files[0]
                result = classify_image(image_path)
                self.classification_output.setPlainText(result)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
