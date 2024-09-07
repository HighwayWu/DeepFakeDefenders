# DeepFakeDefenders

<p align='center'>  
  <img src='https://github.com/HighwayWu/DeepFakeDefenders/blob/main/imgs/competition.png' width='850'/>
</p>

The first-place solution by the "JTGroup" team in the [The Global Multimedia Deepfake Detection (Image Track)](https://www.kaggle.com/competitions/multi-ffdi/overview) competition.

## Table of Contents

- [Motivation](#motivation)
- [Framework](#framework)
- [Inference](#inference)
- [Training](#training)
- [Acknowledgements](#acknowledgements)

If you prefer reading in Chinese, please [click here](https://github.com/HighwayWu/DeepFakeDefenders/blob/main/README-zh.md) to view the Chinese README.

## Motivation
By visualizing the features of the official training and validation set splits, we derived two key insights: 1) The distribution similarity between the training and validation sets makes it challenging to evaluate the model's performance on unseen test sets; 2) The feature types of fake data are richer than those of real data. Based on these insights, we initially used unsupervised clustering to partition the validation set and then enhanced the types of real and fake data based on adversarial learning principles. These insights inspired the development of our generalizable deepfake detection method.

<p align='center'>  
  <img src='https://github.com/HighwayWu/DeepFakeDefenders/blob/main/imgs/motivation.jpg' width='850'/>
</p>

## Framework
The proposed framework consists of two main stages: Data Preparation and Training. In the Data Preparation stage, we enhance the existing dataset by generating new data through image editing and Stable Diffusion (SD) techniques. We then perform clustering to reassemble the dataset, aiming to improve both detection performance and robustness. In the Training stage, we introduce a series of expert models and optimize them using three types of loss functions: $L_{\mathsf{KL}}$, $L_{\mathsf{NCE}}$, and $L_{\mathsf{CE}}$. This multi-loss approach ensures that the model can effectively distinguish between authentic and manipulated images. For more details, please refer to the technical report (to be published on arXiv later).

<p align='center'>  
  <img src='https://github.com/HighwayWu/DeepFakeDefenders/blob/main/imgs/framework.jpg' width='850'/>
</p>

## Inference

### Environment Setup

Basic environment requirements: CUDA 11.1 + Pytorch 1.9.0.

```bash
git clone https://github.com/HighwayWu/DeepFakeDefenders
cd DeepFakeDefenders
pip install -r requirements.txt
```

### Pretrained Weights Download

The pretrained weights can be downloaded from [Baidu Pan](https://pan.baidu.com/s/1hh6Rub60T7UXok5rqACffQ?pwd=gxu5).

After downloading, place the files under `DeepFakeDefenders/weights`.

### Running in Terminal

```bash
>> python infer.py
```

Enter the image path and expect the following result:
```bash
>> demo/img_1_should_be_0.0016829653177410.jpg
>> Prediction of [demo/img_1_should_be_0.0016829653177410.jpg] being Deepfake: 0.001683078
```

### Running with API

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

Example of an HTTP request:
```bash
curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"image": "yourBase64EncodedImage"}'
```

Python request example:
```python
import base64
import requests

# Convert image to Base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_string

# Call the API
api_url = 'http://localhost:8080/predict'
headers = {"Content-Type": "application/json"}
base64_image = encode_image_to_base64("./demo/img_1_should_be_0.0016829653177410.jpg")
payload = {"image": base64_image}

try:
    response = requests.post(api_url, json=payload, headers=headers)
    if response.status_code == 200:
        # Extract the result, return the DeepFake score
        result = response.json().get('result', 0)
        print("Prediction result:", round(result, 3))
    else:
        raise ValueError(f"API call failed, status code: {response.status_code}")
except Exception as e:
    print(f"API call error: {e}")
    
    
    
'''
Prediction result: 0.002
'''
```

### Running with WebUI

The WebUI is separate from the inference, so you need to start the API service on a GPU server first and then modify the `api_url` in `webui/app.py` to your own.

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

Home page:

<p align='center'>  
  <img src='https://github.com/HighwayWu/DeepFakeDefenders/blob/main/imgs/webui_2.ppg' width='850'/>
</p>

Upload prediction:

<p align='center'>  
  <img src='https://github.com/HighwayWu/DeepFakeDefenders/blob/main/imgs/webui_3.ppg' width='850'/>
</p>


## Training
The training code (e.g., unsupervised clustering and joint optimization loss) and detailed technical report will be available shortly.

## Acknowledgements
- This work was partially performed at SICC, supported by SKL-IOTSC, University of Macau.

<p align='left'>  
  <img src='https://github.com/HighwayWu/DeepFakeDefenders/blob/main/imgs/organization.png' width='450'/>
</p>

- Inclusion Conference on the Bund, the organizer of the competition.

## License
This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

For inquiries or to obtain permission for commercial use, please contact Yiming Chen (yc17486@umac.mo).