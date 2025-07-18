from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('/home/lab/digital优化升级/yolo_series/ultralytics-main/runs/detect/train2/weights/best.pt') # 这里用你训练完以后保存的模型文件

# 推理单张图片
results = model('path/to/image.jpg')

# 显示推理结果
for result in results:
	result.show()
