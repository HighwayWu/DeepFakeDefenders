import os
import cv2
import torch.nn as nn
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify
from torchvision import transforms
from network import MFF_MoE  

app = Flask(__name__)

# 设置 GPU 设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 定义模型推理类
class NetInference:
    def __init__(self):
        self.net = MFF_MoE(pretrained=False)  # 使用自定义模型
        self.net.load(path='weights/')  # 从指定路径加载权重
        self.net = nn.DataParallel(self.net).cuda()
        self.net.eval()

        # 定义图像预处理步骤
        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((512, 512), antialias=True),
        ])

    # 推理函数，输入是PIL图像，输出是模型预测结果
    def infer(self, image):
        x = self.transform_val(image).unsqueeze(0).cuda()
        pred = self.net(x)
        pred = pred.detach().cpu().numpy()
        return pred

# 实例化推理模型
model = NetInference()

# 解析Base64图片的函数
def decode_image(base64_str):
    try:
        # Base64解码为二进制数据
        img_data = base64.b64decode(base64_str)
        # 转换为PIL Image格式
        image = Image.open(BytesIO(img_data)).convert('RGB')
        return image
    except Exception as e:
        print(f"解码Base64图片时发生错误: {e}")
        return None

# 定义API的POST路由，处理图像推理请求
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    # 从请求数据中提取Base64图像
    base64_image = data['image']
    
    # 解码图像
    image = decode_image(base64_image)
    if image is None:
        return jsonify({'error': 'Invalid image data'}), 400

    try:
        # 对图像进行推理
        prediction = model.infer(image)
        result = float(prediction[0])  # 取第一个结果，假设这是Deepfake的检测结果
        return jsonify({'result': result}), 200
    except Exception as e:
        print(f"推理时发生错误: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

# 启动Flask服务
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)