import cv2
import os


def detect_eyes(image_path, eye_cascade):
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

        # 检测人脸区域中的眼睛
        eyes = eye_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(eyes) > 0:
            # 在原图上绘制矩形框来标出眼睛
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(image, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)  # 绿色框，线宽2

            print(1)  # 检测到眼睛，输出1
        else:
            print(0)  # 没有检测到眼睛，输出0

        # 只处理第一个人脸，显示框选结果
        break

    # 如果没有检测到任何人脸，输出0
    if len(faces) == 0:
        print(0)

    # 显示框选后的图像
    cv2.imshow('Detected Eyes', image)
    cv2.waitKey(0)  # 等待按键输入
    cv2.destroyAllWindows()


# 加载眼睛检测的Haar Cascade分类器
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 提供图片路径
image_path = r'test.png'  # 使用绝对路径或确保正确的相对路径

# 进行眼睛检测
detect_eyes(image_path, eye_cascade)
