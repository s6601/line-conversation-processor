# LINE 對話紀錄處理工具

這是一個用於處理 LINE 對話錄影的工具，可以自動提取並拼接對話內容，生成易於閱讀的長圖。

## 功能特點

- 自動識別並裁剪 LINE 對話區域
- 智能去除重複內容
- 自動拼接多張圖片
- 生成美觀的 HTML 預覽頁面
- 支援大檔案處理

## 本地部署

1. 克隆專案：
```bash
git clone https://github.com/s6601/line-conversation-processor.git
cd line-conversation-processor
```

2. 安裝依賴：
```bash
pip install -r requirements.txt
```

3. 運行應用：
```bash
python app.py
```

4. 開啟瀏覽器訪問：`http://localhost:3000`

## 使用說明

1. 點擊「選擇檔案」按鈕上傳 LINE 對話錄影
2. 等待處理完成
3. 點擊結果連結查看處理後的對話紀錄

## 技術細節

- 使用 OpenCV 進行圖像處理
- Flask 框架提供 Web 介面
- 智能演算法去除重複內容
- 自動調整圖片大小和比例

## 注意事項

- 建議使用 MP4 格式的影片
- 影片解析度建議至少 720p
- 處理時間取決於影片長度和電腦效能

## 授權

MIT License

## 作者

s6601

## 貢獻

歡迎提交 Issue 和 Pull Request！ 