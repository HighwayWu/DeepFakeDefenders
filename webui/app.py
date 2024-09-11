import requests
import base64
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import random
import uuid
import os
import io


app = Flask(__name__)

# 发送Base64编码的图片到DeepFakeDefenders检测API
def call_deepfake_api(base64_image):
    # 调用API
    api_url = 'http://localhost:8080/predict'  # API 地址
    headers = {"Content-Type": "application/json"}
    payload = {"image": base64_image}
    
    try:
        response = requests.post(api_url, json=payload, headers=headers)
        if response.status_code == 200:
            # 提取结果，返回DeepFake分数
            result = response.json().get('result', 0)
            return round(result, 3)  # 返回保留3位小数的结果
        else:
            raise ValueError(f"API调用失败，状态码：{response.status_code}")
    except Exception as e:
        print(f"API调用出错: {e}")
        return None


def generate_certificate(score, image, background_color='white'):
    w = 700
    h = 900
    # 创建背景图片，可以根据传入参数设定背景颜色
    if background_color == 'blue':
        cert_img = Image.new('RGB', (w, h), color='#E0FFFF')  # 水蓝色
    elif background_color == 'green':
        cert_img = Image.new('RGB', (w, h), color='#7CFC00')  # 渐变荧光绿
    else:
        cert_img = Image.new('RGB', (w, h), color='white')  # 白色背景

    draw = ImageDraw.Draw(cert_img)
    font_path = "./static/font/SimHei.ttf"
    # 字体设置
    try:
        title_font = ImageFont.truetype(font_path, 40)  # 标题字体
        text_font = ImageFont.truetype(font_path, 25)   # 正文字体
        bold_font = ImageFont.truetype(font_path, 25)  # 加粗的字体
        small_font = ImageFont.truetype(font_path, 20)  # 小字体
        italic_font = ImageFont.truetype(font_path, 20)  # 辅助说明字体（斜体）
    except IOError:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        bold_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
        italic_font = ImageFont.load_default()

    # 证书标题
    title_text = "DeepFakeDefenders 真假预测"
    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    text_width = title_bbox[2] - title_bbox[0]
    draw.text(((w - text_width) / 2, 50), title_text, font=title_font, fill="black")

    # 分数信息
    score_text = "您上传的图片真假预测得分为："
    score_bbox = draw.textbbox((0, 0), score_text, font=text_font)
    score_width = score_bbox[2] - score_bbox[0]
    draw.text(((w - score_width) / 2, 150), score_text, font=text_font, fill="black")

    # 加粗分数
    score_value = f"{score:.2f}"
    score_value_bbox = draw.textbbox((0, 0), score_value, font=bold_font)
    score_value_width = score_value_bbox[2] - score_value_bbox[0]
    draw.text(((w - score_value_width) / 2, 210), score_value, font=bold_font, fill="black")

    # 加载并显示缩略图
    uploaded_image = Image.open(image)
    uploaded_image.thumbnail((200, 200))  # 调整缩略图大小
    image_x = (w - 200) // 2  # 图片居中显示
    cert_img.paste(uploaded_image, (image_x, 280))

    # 小字说明
    small_text = "(结果仅供参考，得分越高越有可能是假图片)"
    small_text_bbox = draw.textbbox((0, 0), small_text, font=italic_font)
    small_text_width = small_text_bbox[2] - small_text_bbox[0]
    draw.text(((w - small_text_width) / 2, 520), small_text, font=italic_font, fill="gray")

    # 致谢部分
    thanks_text = "致谢"
    thanks_text_bbox = draw.textbbox((0, 0), thanks_text, font=bold_font)
    thanks_text_width = thanks_text_bbox[2] - thanks_text_bbox[0]
    draw.text(((w - thanks_text_width) / 2, 620), thanks_text, font=bold_font, fill="black", spacing=18)

    details_text = (
        '本服务使用"JTGroup"团队开源项目\n'
        'DeepFakeDefenders 推理得到结果，感谢该团队成员的开源分享。'
    )
    
    # 多行文本的居中显示
    draw.multiline_text((50, 680), details_text, font=small_font, fill="black", align="left", spacing=18)

    # 编号和创建日期
    cert_number = f"证书编号：{random.randint(1000, 9999)}"
    date_created = f"创建日期：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    # 右下角显示编号和日期
    draw.text((300, 800), cert_number, font=small_font, fill="black")
    draw.text((300, 840), date_created, font=small_font, fill="black")

    # 保存证书到内存
    cert_io = io.BytesIO()
    cert_img.save(cert_io, 'PNG')
    cert_io.seek(0)

    # 将二进制数据转换为Base64字符串
    base64_cert = base64.b64encode(cert_io.getvalue()).decode('utf-8')

    return base64_cert




# 主页路由
@app.route('/')
def index():
    return render_template('index.html')

# 图片上传并检测真假
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "没有上传文件"}), 400

    file = request.files['image']
    background_color = request.form.get('background_color', 'white')  # 获取背景颜色

    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400

    if file:
        # 将文件读取为字节流并转换为 Base64 编码
        base64_image = base64.b64encode(file.read()).decode('utf-8')

        # 调用DeepFakeDefenders接口
        score = call_deepfake_api(base64_image)

        if score is None:
            return jsonify({"error": "检测失败，请稍后再试"}), 500

        # 生成证书图片并返回Base64编码
        file.seek(0)  # 重置文件指针
        base64_cert = generate_certificate(score, file, background_color)

        # 返回Base64编码的证书图片
        return jsonify({"message": "检测成功", "certificate": base64_cert})



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=1234)

