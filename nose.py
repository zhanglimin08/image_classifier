import cv2
import os


def detect_nose(image_path, nose_cascade):
    # 检查文件路径是否存在
    if not os.path.exists(image_path):
        print(f"文件 '{image_path}' 不存在，请检查路径！")
        return

    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print("图像无法加载，请检查文件是否损坏或路径错误。")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像

    # 使用人脸检测器来检测图像中的人脸
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 遍历检测到的每个人脸
    for (x, y, w, h) in faces:
        # 提取人脸区域
        face_region = gray[y:y + h, x:x + w]

        # 检测人脸区域中的鼻子
        nose = nose_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(nose) > 0:
            # 在原图上绘制矩形框来标出鼻子
            for (mx, my, mw, mh) in nose:
                cv2.rectangle(image, (x + mx, y + my), (x + mx + mw, y + my + mh), (0, 255, 0), 2)  # 绿色框，线宽2
            print(1)  # 检测到鼻子，输出1
        else:
            print(0)  # 没有检测到鼻子，输出0

        # 只处理第一个人脸，显示框选结果
        break

    # 如果没有检测到任何人脸，输出0
    if len(faces) == 0:
        print(0)

    # 显示框选后的图像
    cv2.imshow('Detected Nose', image)
    cv2.waitKey(0)  # 等待按键输入
    cv2.destroyAllWindows()


# 加载鼻子检测的Haar Cascade分类器
nose_cascade_path = 'haarcascade_mcs_nose.xml'  # 修改为实际路径
nose_cascade = cv2.CascadeClassifier(nose_cascade_path)


# 提供图片路径
image_path = r'test.png'  # 使用绝对路径或确保正确的相对路径

# 进行鼻子检测
detect_nose(image_path, nose_cascade)
