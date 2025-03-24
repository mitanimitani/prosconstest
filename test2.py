import cv2
import numpy as np

def extract_metal_panel(image_path):
    # 画像の読み込み
    image = cv2.imread(image_path)
    if image is None:
        print("画像が見つかりません。")
        return
    # 画像のリサイズ
    image = cv2.resize(image, (1000, 800))

    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # パネル穴部検出
    edges = cv2.Canny(gray, 300, 600)
    mask = np.zeros_like(gray)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    mask = cv2.medianBlur(mask, 5)
    _, mask = cv2.threshold(mask,110, 255, cv2.THRESH_BINARY)
    
    # ガウシアンフィルタを適用（縦方向ノイズ除去）
    gray = cv2.GaussianBlur(gray, (15, 5), 0)
    gray = cv2.GaussianBlur(gray, (15, 5), 0)
    
    # ２値化
    _, binary = cv2.threshold(gray,110, 255, cv2.THRESH_BINARY)

    # パネル外形検出
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = 5000
    
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            cv2.drawContours(binary, [contour], -1, (255), thickness=cv2.FILLED)
    
    binary_inv = cv2.bitwise_not(binary)

    contours, _ = cv2.findContours(binary_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            cv2.drawContours(binary_inv, [contour], -1, (255), thickness=cv2.FILLED)    
    
    contours, _ = cv2.findContours(binary_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            cv2.drawContours(binary_inv, [contour], -1, (0), thickness=cv2.FILLED)
    
    #パネル穴部と合成
    temp = cv2.bitwise_or(binary_inv, mask)
    mask = cv2.bitwise_not(temp)

    # マスクを適用して金属パネル部分のみ抽出
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # 画像の表示
    cv2.imshow('Original Image', image)
    cv2.imshow('Extracted Metal Panel', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 画像処理を実行
extract_metal_panel("metal_panel.jpg")