import torch
from torch import nn
import torchvision
from torchvision import transforms
import numpy as np
import os
import face_recognition
import argparser

parser = argparse.ArgumentParser(description="Predict age given a face")
parser.add_argument('img_path', type=str, help="path to the image")
parser.add_argument('params_path', type=str, help="path to the model's parameters")

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Block_prototype_1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Block_prototype_1, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3,3), padding=0)
        self.relu = nn.ReLU()
        self.batchNorm = nn.BatchNorm2d(num_features=out_channel)
        self.max_pool = nn.MaxPool2d((2, 2))
    

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.batchNorm(out)
        out = self.max_pool(out)
        return out
    
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.in_channel = 1
        self.batchNorm = nn.BatchNorm2d(self.in_channel)
        self.block1 = Block_prototype_1(self.in_channel, self.in_channel * 4)
        self.block2 = Block_prototype_1(self.in_channel * 4, self.in_channel * 32)
        self.block3 = Block_prototype_1(self.in_channel * 32, self.in_channel * 64)
        self.block4 = Block_prototype_1(self.in_channel * 64, self.in_channel * 32)
        self.linear1 = nn.Linear(2592, 512)
        self.drop_out1 = nn.Dropout(p=0.4)
        self.linear2 = nn.Linear(512, 256)
        self.drop_out2 = nn.Dropout(p=0.3)
        self.linear3 = nn.Linear(256, 1)
  
    def forward(self, x):
        batch_size = x.shape[0]
        out = self.batchNorm(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out_flatten = out.view(batch_size, -1)
        out = self.linear1(out_flatten)
        out = self.drop_out1(out)
        out = self.linear2(out)
        out = self.drop_out2(out)
        out = self.linear3(out)

        return out
    
model = Net1().to(device=device)
model.load_state_dict(torch.load(parser.params.path))

img = torchvision.io.read_image(parser.img_path)
image = face_recognition.load_image_file(parser.img_path)
face_locations = face_recognition.face_locations(image)
if len(face_locations) == 0:
    raise Exception("No face detected")
if face_locations[0] > face_locations[2]:
    x1, x2 = face_locations[2], face_locations[0]
else:
    x1, x2 = face_locations[0], face_locations[2]

if face_locations[1] > face_locations[3]:
    y1, y2 = face_locations[3], face_locations[1]
else:
    y1, y2 = face_locations[1], face_locations[3]
    
img_face = image[:, x1:x2, y1:y2]
img_face = img_face.unsqueeze(0).to(device=device, dtype=torch.float)
pipe = transforms.Compose([transforms.Grayscale(1),
                           transforms.Resize((180, 180))])
img_face = pipe(img_face).unsqueeze(0)

model.eval()
prediction = model(img_face).item()
prediction = prediction if prediction >= 0 else 0

print("Prediction %d" %(prediction))

print("Preprocessing Face")
print("Prediction: %d" %(prediction))
