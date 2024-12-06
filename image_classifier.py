import cv2
import os
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
import joblib

# 眼睛检测
def detect_eyes(image, eye_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_region = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(eyes) > 0:
            return 1  # 检测到眼睛
    return 0  # 没有检测到眼睛


# 头发检测
def load_svm_model(model_path):
    """加载训练好的 SVM 模型"""
    return joblib.load(model_path)


def preprocess_image(image_path):
    """预处理图片：读取图片，调整大小，并转换为灰度图"""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (230, 230))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def extract_hog_features(image):
    """提取图像的HOG特征"""
    features, _ = hog(image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    return features


def detect_hair(image_path, svm_model):
    """使用 SVM 模型来检测图片是否有头发"""
    gray_image = preprocess_image(image_path)
    hog_features = extract_hog_features(gray_image)
    prediction = svm_model.predict([hog_features])
    return prediction[0]  # 返回 1 或 0


# 嘴巴检测
def detect_mouth(image, mouth_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_region = gray[y:y + h, x:x + w]
        mouths = mouth_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(mouths) > 0:
            return 1  # 检测到嘴巴
    return 0  # 没有检测到嘴巴

# 鼻子检测
def detect_mouth(image, nose_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_region = gray[y:y + h, x:x + w]
        nose = nose_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(nose) > 0:
            return 1  # 检测到鼻子
    return 0  # 没有检测到鼻子



# 主程序
def main():
    # 设置路径
    image_folder = r"D:\Code\Image classifier\test"  # 请确保路径正确
    model_path = 'D:\Code\Image classifier\hair_HOG.pkl'  # 请替换为实际的SVM模型路径
    mouth_cascade_path = 'D:\Code\Image classifier\haarcascade_mcs_mouth.xml'  # 请替换为实际路径
    nose_cascade_path = 'D:\Code\Image classifier\haarcascade_mcs_nose.xml'  # 请替换为实际路径

    # 加载 SVM 模型
    svm_model = load_svm_model(model_path)
    # 加载嘴巴检测的Haar Cascade分类器
    mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)
    # 加载眼睛检测的Haar Cascade分类器
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    # 加载鼻子检测的Haar Cascade分类器
    nose_cascade = cv2.CascadeClassifier(nose_cascade_path)

    # 打印标题行
    print(" Image a1 a2 a3 a4")

    # 遍历图片文件夹中的所有图片文件
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)

        # 确保只处理图片文件
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                continue  # 如果图片无法读取，则跳过

            # 检测眼睛
            eyes_detected = detect_eyes(image, eye_cascade)

            # 检测嘴巴
            mouth_detected = detect_mouth(image, mouth_cascade)

            # 检测头发
            hair_detected = detect_hair(image_path, svm_model)

            # 检测鼻子
            nose_detected = detect_mouth(image, nose_cascade)

            # 格式化输出
            print(f"{filename} {hair_detected}  {mouth_detected}  {eyes_detected}  {nose_detected}")


if __name__ == "__main__":
    main()
