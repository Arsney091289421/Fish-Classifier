import os
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# 让 Flask 确保 `templates/` 目录的路径正确
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在的目录
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")  # 计算 templates 的完整路径

app = Flask(__name__, template_folder=TEMPLATE_DIR)  # 使用绝对路径

# 加载模型、标准化工具和标签编码器
model = joblib.load(os.path.join(BASE_DIR, "fish_species_classifier.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
le = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

@app.route('/')
def home():
    return render_template('index.html')  # 确保 index.html 存在

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(request.form.get(key)) for key in ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]]
        data_scaled = scaler.transform([data])  
        prediction = model.predict(data_scaled) 
        predicted_species = le.inverse_transform(prediction)[0]  

        return jsonify({"prediction": predicted_species})  

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
