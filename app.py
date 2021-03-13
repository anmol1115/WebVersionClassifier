import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image

from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify

import io
import base64
from PIL import Image

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = t.reshape(-1, 12*4*4)
        t = F.relu(self.fc1(t))

        t = F.relu(self.fc2(t))
        t = F.softmax(self.out(t), dim=1)
        return t

    def save(self):
        torch.save(self.state_dict(), './data/model.pth')

def process(img):
    img = img.resize((28, 28))
    img = transforms.ToTensor()(img)
    img = img[3].unsqueeze(0).unsqueeze(0)
    return img

app = Flask(__name__)
network = Network()
network.load_state_dict(torch.load('./data/model.pth'))
network.eval()

@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        message = request.get_json(force=True)
        encoded_image = message['image']
        decoded_image = base64.b64decode(encoded_image)
        image = Image.open(io.BytesIO(decoded_image))
        image_tensor = process(image)
        pred = network(image_tensor).squeeze(1).tolist()

        prob = {
            'zero': pred[0][0],
            'one': pred[0][1],
            'two': pred[0][2],
            'three': pred[0][3],
            'four': pred[0][4],
            'five': pred[0][5],
            'six': pred[0][6],
            'seven': pred[0][7],
            'eight': pred[0][8],
            'nine': pred[0][9]
        }

        return jsonify(prob)

if __name__ == '__main__':
    app.run(debug=True)