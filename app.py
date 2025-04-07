import cv2
import numpy as np
import base64
import tempfile
import time
from flask import Flask, request, render_template_string

app = Flask(__name__)

def crop_chat_area(frame):
    """裁剪 LINE 聊天內容區域，排除狀態欄和輸入欄"""
    height, width = frame.shape[:2]
    top_crop = int(height * 0.1)
    bottom_crop = int(height * 0.85)
    return frame[top_crop:bottom_crop, :]

def calculate_pixel_diff(img1, img2):
    """計算兩張圖片的像素差異"""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    diff = cv2.absdiff(img1, img2)
    return np.mean(diff)

def find_overlap_and_crop_template(prev_img, new_img, template_height=50, match_threshold=0.9):
    """利用模板匹配偵測 prev_img 與 new_img 的重疊區，並裁剪掉重疊部分。"""
    if prev_img.shape[1] != new_img.shape[1]:
        new_img = cv2.resize(new_img, (prev_img.shape[1], new_img.shape[0]))

    template = prev_img[-template_height:, :]
    search_height = min(new_img.shape[0], template_height * 2)
    search_region = new_img[:search_height, :]

    res = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val >= match_threshold:
        crop_y = max_loc[1] + template_height
        cropped_new = new_img[crop_y:, :]
    else:
        cropped_new = new_img
    return cropped_new

def extract_and_stitch_frames(video_path, max_height=20000, diff_threshold=20):
    """
    提取影片中的關鍵幀，利用模板匹配去除相鄰幀重疊區，
    並垂直拼接成長圖，回傳「每個段落」對應的 base64 圖片清單。
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps // 2)

    frames = []
    frame_count = 0
    last_added_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_cropped = crop_chat_area(frame)
        # 固定寬度 800 等比例縮放
        frame_cropped = cv2.resize(
            frame_cropped,
            (800, int(800 * frame_cropped.shape[0] / frame_cropped.shape[1])),
            interpolation=cv2.INTER_AREA
        )

        if frame_count % frame_interval == 0:
            if last_added_frame is not None:
                diff_score = calculate_pixel_diff(last_added_frame, frame_cropped)
                if diff_score > diff_threshold:
                    frames.append(frame_cropped)
                    last_added_frame = frame_cropped
                else:
                    frame_interval = max(1, frame_interval // 2)
            else:
                frames.append(frame_cropped)
                last_added_frame = frame_cropped
        else:
            frame_interval = min(fps * 2, frame_interval + 1)

        frame_count += 1

    cap.release()

    if not frames:
        return None

    stitched_images_base64 = []
    stitched_part = frames[0]

    for idx in range(1, len(frames)):
        new_frame_cropped = find_overlap_and_crop_template(stitched_part, frames[idx],
                                                           template_height=50,
                                                           match_threshold=0.9)
        stitched_part = np.vstack([stitched_part, new_frame_cropped])

        if stitched_part.shape[0] >= max_height:
            # 儲存當前的 stitched_part 成 base64
            stitched_images_base64.append(np_img_to_base64(stitched_part))
            # 開新一段
            stitched_part = new_frame_cropped

    if stitched_part is not None:
        stitched_images_base64.append(np_img_to_base64(stitched_part))

    return stitched_images_base64

def np_img_to_base64(np_img):
    """把 numpy 圖片轉成 base64 字串，前端可用 <img src="data:image/jpg;base64, ..."> 顯示"""
    _, buffer = cv2.imencode('.jpg', np_img)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str

def generate_html(base64_list):
    """直接動態產生整個 HTML 字串，內嵌 base64 圖片"""
    html_content = """
    <html>
    <head>
        <title>LINE Conversation Record</title>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .stitched-image { margin: 20px 0; }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
    <h1>LINE 對話紀錄 - 拼接圖</h1>
    <p>以下為從影片中提取並拼接的對話紀錄，利用模板匹配方式消除重疊區。</p>
    """

    for i, b64_img in enumerate(base64_list):
        html_content += f"""
        <div class="stitched-image">
            <h2>段落 {i + 1}</h2>
            <img src="data:image/jpg;base64,{b64_img}" alt="Stitched Image {i + 1}"/>
        </div>
        """

    html_content += """
    </body>
    </html>
    """
    return html_content

# 前端首頁 HTML 表單
INDEX_HTML = """
<!doctype html>
<html>
<head>
    <title>影片上傳</title>
</head>
<body>
    <h1>上傳影片進行 LINE 對話紀錄處理 (一次性，不儲存檔案)</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <p>
            <input type="file" name="video_file" accept="video/*" required>
        </p>
        <p>
            <input type="submit" value="上傳並處理">
        </p>
    </form>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(INDEX_HTML)

@app.route('/upload', methods=['POST'])
def upload():
    if 'video_file' not in request.files:
        return "沒有檔案上傳", 400

    file = request.files['video_file']
    if file.filename == '':
        return "檔案名稱空白", 400

    # 讀取整個檔案到記憶體
    file_data = file.read()
    if not file_data:
        return "上傳檔案為空", 400

    # 用 NamedTemporaryFile 供 OpenCV 讀取 (臨時檔, 關閉後自動刪除)
    with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp:
        tmp.write(file_data)
        tmp.seek(0)
        video_path = tmp.name

        # 進行拼接
        stitched_base64_list = extract_and_stitch_frames(video_path)
        if not stitched_base64_list:
            return "處理影片失敗或無法提取任何幀", 500

    # 產生 HTML
    result_html = generate_html(stitched_base64_list)
    return render_template_string(result_html)

if __name__ == "__main__":
    app.run(port=3000, debug=True)
