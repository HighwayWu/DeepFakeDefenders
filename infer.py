import os
import cv2
import random
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from network import MFF_MoE

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--local_weight', type=str, default='weights/', help='trained weights path')
args = parser.parse_args()


class NetInference():
    def __init__(self):
        self.net = MFF_MoE(pretrained=False)
        self.net.load(path=args.local_weight)
        self.net = nn.DataParallel(self.net).cuda()
        self.net.eval()
        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((512, 512), antialias=True),
        ])

    def infer(self, input_path=''):
        x = cv2.imread(input_path)[..., ::-1]
        x = Image.fromarray(np.uint8(x))
        x = self.transform_val(x).unsqueeze(0).cuda()
        pred = self.net(x)
        pred = pred.detach().cpu().numpy()
        return pred


if __name__ == '__main__':
    model = NetInference()
    while True:
        print('Please input the image path:')
        input_path = input()
        try:
            res = model.infer(input_path)
            print('Prediction of [%s] being Deepfake: %10.9f' % (input_path, res))
        except:
            print('Error: Image [%s]' % input_path)
