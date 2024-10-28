import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# 加载预训练的YOLOv11姿态估计模型
model = YOLO('yolo11n-pose.pt')

# 初始化摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLO模型进行推理
    results = model(frame)

    # 检查是否有检测结果
    if results and len(results) > 0:
        # 假设第一个检测结果是我们感兴趣的人
        person_results = results[0]

        # 检查是否有关键点检测结果
        if person_results.keypoints.xy is not None and person_results.keypoints.conf is not None:
            # 提取关键点坐标和置信度
            keypoints_xy = person_results.keypoints.xy.numpy()  # 获取关键点坐标
            keypoints_conf = person_results.keypoints.conf.numpy()  # 获取关键点置信度

            # 检查是否检测到人
            person_detected = keypoints_xy.shape[1] > 0

            # 初始化要显示的消息
            message = "怎么走了？滚来学习！"
            message_color = (0, 0, 255)

            # 遍历所有检测到的关键点
            for i, (x, y) in enumerate(keypoints_xy[0]):
                if keypoints_conf[0][i] > 0.5:  # 假设conf > 0.5表示关键点置信度足够
                    # 根据检测到的姿态更新消息
                    if i == 0:  # 假设关键点0是头部
                        if y < frame.shape[0] * 0.2:
                            message = "不错，继续保持！"
                            message_color = (0, 255, 0)
                        else:
                            message = "抬头！抬头！抬头！"
                            message_color = (255, 0, 0)


            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype("simhei.ttf", 60) 
            draw.text((10, 30), message, font=font, fill=message_color)

            # 将Pillow图像转换回OpenCV格式
            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 显示结果
    cv2.imshow('Pose Estimation', frame)

    # 按'q'退出
    if cv2.waitKey(1) == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()