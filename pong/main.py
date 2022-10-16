
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)   # 打开摄像头
cap.set(3, 1280)         # 设定摄像画面尺寸
cap.set(4, 720)

# 导入背景，结束画面，球和左右人物
imgBackground = cv2.imread("Resources/Background.png")
imgGameOver = cv2.imread("Resources/gameOver.png")
imgBall = cv2.imread("Resources/Ball.png", cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread("Resources/bat1.png", cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread("Resources/bat2.png", cv2.IMREAD_UNCHANGED)

# 手检测
detector = HandDetector(detectionCon=0.8, maxHands=2)   # detectionCon： 手部检测模型的最小置信值（0-1之间），超过阈值则检测成功。默认为 0.5

# 变量
ballPos = [100, 100]       # 球初始位置
speedX = 25                # 初始水平速度
speedY = 25                # 初始垂直速度
gameOver = False
score = [0, 0]             # 双方得分

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    imgRaw = img.copy()

    # 识别获得hands手的信息和img绘图
    hands, img = detector.findHands(img, flipType=False)

    # 将检测图像与背景融合
    img = cv2.addWeighted(img, 0.1, imgBackground, 0.9, 0)

    # 获得手的位置
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = imgBat1.shape
            y1 = y - h1 // 2
            y1 = np.clip(y1, 20, 415)
    # 检测击球
            if hand['type'] == "Left":
                img = cvzone.overlayPNG(img, imgBat1, (59, y1))
                if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    speedX += 5   # 击球后逐次加速
                    ballPos[0] += 30
                    score[0] += 1

            if hand['type'] == "Right":
                img = cvzone.overlayPNG(img, imgBat2, (1195, y1))
                if 1195 - 50 < ballPos[0] < 1195 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    speedX -= 5
                    ballPos[0] -= 30
                    score[1] += 1

    # 游戏结束
    if ballPos[0] < 40 or ballPos[0] > 1200:
        gameOver = True
    # 结束画面
    if gameOver:
        img = imgGameOver
        cv2.putText(img, str(score[1] + score[0]).zfill(2), (585, 460), cv2.FONT_HERSHEY_COMPLEX,
                    2.5, (200, 0, 200), 5)

    # 球的运动
    else:

        if ballPos[1] >= 500 or ballPos[1] <= 10:
            speedY = -speedY

        ballPos[0] += speedX
        ballPos[1] += speedY

        # 在画面上显示球
        img = cvzone.overlayPNG(img, imgBall, ballPos)
        # 两侧显示得分
        cv2.putText(img, str(score[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
        cv2.putText(img, str(score[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)

    # 左下角显示摄像头画面
    img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    # 按r键重新开始
    if key == ord('r'):
        ballPos = [100, 100]
        speedX = 25
        speedY = 25
        gameOver = False
        score = [0, 0]
