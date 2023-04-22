from flask import Flask, request, render_template
import torch
import numpy as np

app = Flask(__name__)

# Build model
class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.upconv1 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.upconv2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv5 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv3 = torch.nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)
        self.conv6 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = self.pool3(x)
        x = self.upconv1(x)
        x = self.conv4(x)
        x = torch.nn.functional.relu(x)
        x = self.upconv2(x)
        x = self.conv5(x)
        x = torch.nn.functional.relu(x)
        x = self.upconv3(x)
        x = self.conv6(x)
        x = self.sigmoid(x)
        return x

model = NeuralNet()

# Load model weights
model.load_state_dict(torch.load('model_weights.pt'))

# Set model to evaluation mode
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    video_file = request.files['video']
    # Process the video file using the trained model
    # Return the processed video file to the user
    return render_template('result.html')

if __name__ == '__main__':
    app.run