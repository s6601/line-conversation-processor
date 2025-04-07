import cv2
import os
from pathlib import Path
import numpy as np
import shutil
import time
from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string

app = Flask(__name__)

# 設定上傳影片與結果存放的資料夾
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


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
    """
    利用模板匹配偵測 prev_img 底部與 new_img 頂部的重疊區域，
    並裁剪掉新圖片中的重疊部分。
    """
    if prev_img.shape[1] != new_img.shape[1]:
        new_img = cv2.resize(new_img, (prev_img.shape[1], new_img.shape[0]))

    # 以 prev_img 底部取出模板
    template = prev_img[-template_height:, :]
    # 在 new_img 的頂部定義搜尋區（取 template_height*2 高度）
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


def extract_and_stitch_frames(video_path, output_dir, max_height=20000, diff_threshold=20):
    """
    提取影片中的關鍵幀，利用模板匹配方式去除相鄰幀的重疊區，
    並垂直拼接成長圖，最終輸出圖片存放在 output_dir 內。
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        # 固定寬度為 800，等比例縮放
        frame_cropped = cv2.resize(frame_cropped,
                                   (800, int(800 * frame_cropped.shape[0] / frame_cropped.shape[1])),
                                   interpolation=cv2.INTER_AREA)

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
        print("未提取到任何幀！")
        return None

    stitched_images = []
    stitched_part = frames[0]

    for idx in range(1, len(frames)):
        new_frame_cropped = find_overlap_and_crop_template(stitched_part, frames[idx],
                                                           template_height=50,
                                                           match_threshold=0.9)
        stitched_part = np.vstack([stitched_part, new_frame_cropped])

        if stitched_part.shape[0] >= max_height:
            seg_path = output_dir / f"stitched_{len(stitched_images)}.jpg"
            cv2.imwrite(str(seg_path), stitched_part)
            stitched_images.append(str(seg_path))
            stitched_part = new_frame_cropped

    if stitched_part is not None:
        seg_path = output_dir / f"stitched_{len(stitched_images)}.jpg"
        cv2.imwrite(str(seg_path), stitched_part)
        stitched_images.append(str(seg_path))

    return stitched_images


def export_to_html(stitched_images, output_file):
    """
    將拼接好的圖片生成 HTML 檔案，
    HTML 中以相對路徑引用 output_frames 資料夾內的圖片。
    """
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
    for i, img_path in enumerate(stitched_images):
        img_name = Path(img_path).name
        html_content += f"""
        <div class="stitched-image">
            <h2>段落 {i + 1}</h2>
            <img src="output_frames/{img_name}" alt="Stitched Image {i + 1}">
        </div>
        """
    html_content += """
    </body>
    </html>
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return output_file


# 前端首頁 HTML 表單
INDEX_HTML = """
<!doctype html>
<html>
<head>
    <title>影片上傳</title>
</head>
<body>
    <h1>上傳影片進行 LINE 對話紀錄處理</h1>
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

    # 產生唯一識別碼，並建立對應結果資料夾
    result_id = str(int(time.time()))
    result_dir = os.path.join(RESULT_FOLDER, result_id)
    os.makedirs(result_dir, exist_ok=True)
    output_frames_dir = os.path.join(result_dir, "output_frames")
    os.makedirs(output_frames_dir, exist_ok=True)

    # 儲存上傳的影片
    video_filename = f"video_{result_id}.mp4"
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    file.save(video_path)

    # 處理影片：提取並拼接圖片
    stitched_images = extract_and_stitch_frames(video_path, output_dir=output_frames_dir)
    if not stitched_images:
        return "處理影片失敗", 500

    # 產生結果 HTML，並存放於結果資料夾下
    html_output_file = os.path.join(result_dir, "conversation.html")
    export_to_html(stitched_images, html_output_file)

    # 返回結果 HTML 的完整網址
    result_url = url_for('results_file', result_id=result_id, filename="conversation.html", _external=True)
    return f"處理完成，請點擊連結查看結果：<a href='{result_url}' target='_blank'>{result_url}</a>"


@app.route('/results/<result_id>/<path:filename>')
def results_file(result_id, filename):
    directory = os.path.join(RESULT_FOLDER, result_id)
    return send_from_directory(directory, filename)


if __name__ == "__main__":
    app.run(port=3000, debug=True)
