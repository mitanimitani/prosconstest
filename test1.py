import cv2
import numpy as np

# 画像を読み込み、表示
image = cv2.imread("milkdrop.bmp")
cv2.imshow("Original Image", image)

# グレースケール変換 & 2値化（閾値処理）
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 133, 255, cv2.THRESH_BINARY)

# 輪郭抽出
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ノイズ除去（一定以上の面積を持つ領域のみ選択）
min_area = 2500  # ノイズ除去のための面積閾値
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# マスク作成
mask = np.zeros_like(gray)
cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

# マスクを適用してミルククラウンのみを抽出
result = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Milk Crown Extracted", result)

cv2.waitKey(0)
cv2.destroyAllWindows()