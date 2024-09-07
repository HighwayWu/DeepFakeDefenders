# DeepFakeDefenders

<p align='center'>  
  <img src='https://github.com/HighwayWu/DeepFakeDefenders/blob/main/imgs/competition.png' width='850'/>
</p>

[The Global Multimedia Deepfake Detection (Image Track)](https://www.kaggle.com/competitions/multi-ffdi/overview)比赛中，由"JTGroup"团队获得的第一名方案。

## 目录

- [动机](#动机)
- [框架](#框架)
- [推理](#推理)
- [训练](#训练)
- [致谢](#致谢)

## 动机
通过对官方训练集和验证集划分的特征可视化分析，我们得出两个关键见解：1) 训练集和验证集的分布相似，使得评估模型在未见过的测试集上的表现具有挑战性；2) 伪造数据的特征类型比真实数据更加丰富。基于这些见解，我们首先使用无监督聚类对验证集进行划分，然后基于对抗学习原则增强真实和伪造数据的类型。这些见解促使我们开发了一个通用的深度伪造检测方法。

<p align='center'>  
  <img src='https://github.com/HighwayWu/DeepFakeDefenders/blob/main/imgs/motivation.jpg' width='850'/>
</p>

## 框架
我们提出的方法框架包括两个主要阶段：数据准备和训练。在数据准备阶段，我们通过图像编辑和Stable Diffusion（SD）技术生成新数据来增强现有数据集。然后，我们进行聚类以重新组合数据集，以提高检测性能和方法的鲁棒性。在训练阶段，我们引入了一系列专家模型，并使用三种损失函数进行优化：$L_{\mathsf{KL}}$，$L_{\mathsf{NCE}}$，和$L_{\mathsf{CE}}$。这种多损失方法确保模型能够有效区分真实和伪造图像。有关详细信息，请参阅技术报告（稍后将在arXiv上发布）。

<p align='center'>  
  <img src='https://github.com/HighwayWu/DeepFakeDefenders/blob/main/imgs/framework.jpg' width='850'/>
</p>

## 推理运行

### 环境安装

基础环境要求：CUDA 11.1 + Pytorch 1.9.0 。

```bash
git clone https://github.com/HighwayWu/DeepFakeDefenders
cd DeepFakeDefenders
pip install -r requirements.txt
```
### 预训练权重下载

预训练权重可以从[Baidu Pan](https://pan.baidu.com/s/1hh6Rub60T7UXok5rqACffQ?pwd=gxu5)下载。

下载后文件请放到`DeepFakeDefenders/weights`下。

### 终端运行

```bash
>> python infer.py
```
输入图像路径，预期结果如下：
```bash
>> demo/img_1_should_be_0.0016829653177410.jpg
>> Prediction of [demo/img_1_should_be_0.0016829653177410.jpg] being Deepfake: 0.001683078
```

### API 运行

```bash
python infer_api.py

'''
 * Serving Flask app 'infer_api'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8080
 * Running on http://xxx.xx.x.x:8080
Press CTRL+C to quit
'''
```

http 请求示例：
```bash
curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"image": "你的Base64编码图像"}'
```

python 请求示例：
```python
import base64
import requests

# 将图片转换为 Base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_string

# 调用API
api_url = 'http://localhost:8080/predict'
headers = {"Content-Type": "application/json"}
base64_image = encode_image_to_base64("./demo/img_1_should_be_0.0016829653177410.jpg")
payload = {"image": base64_image}

try:
    response = requests.post(api_url, json=payload, headers=headers)
    if response.status_code == 200:
        # 提取结果，返回DeepFake分数
        result = response.json().get('result', 0)
        print("预测结果：", round(result, 3))
    else:
        raise ValueError(f"API调用失败，状态码：{response.status_code}")
except Exception as e:
    print(f"API调用出错: {e}")
    
    
    
'''
预测结果： 0.002
'''
```

### WebUI 运行

WebUI 和推理是分开的，所以需要先在一台GPU服务器运行启动 API 服务，然后修改`webui/app.py`中`api_url`改成您自己的。

<p align='center'>  
  <img src='https://github.com/HighwayWu/DeepFakeDefenders/blob/main/imgs/webui_1.ppg' width='850'/>
</p>

```bash
cd webui
python app.py 

'''
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:1234
 * Running on http://xxx.xx.x.x:1234
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 108-450-973
'''
```

首页：

<p align='center'>  
  <img src='https://github.com/HighwayWu/DeepFakeDefenders/blob/main/imgs/webui_2.ppg' width='850'/>
</p>

上传预测:

<p align='center'>  
  <img src='https://github.com/HighwayWu/DeepFakeDefenders/blob/main/imgs/webui_3.ppg' width='850'/>
</p>

## 训练
训练代码（例如，无监督聚类和联合优化损失）及详细技术报告将很快发布。

## 致谢
- 本工作部分在澳门大学物联网国家重点实验室（SKL-IOTSC）支持下的SICC进行。

<p align='left'>  
  <img src='https://github.com/HighwayWu/DeepFakeDefenders/blob/main/imgs/organization.png' width='450'/>
</p>

- 包括·外滩大会，比赛的组织者。

## 许可证
此作品受[知识共享署名-非商业性使用 4.0 国际许可协议][cc-by-nc]的保护。

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

如需商业用途许可，请联系陈艺明 (yc17486@umac.mo)。