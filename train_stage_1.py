import sys
sys.path.append("..")

from ultralytics import YOLO
# s 181 layers, 9,429,340 parameters, 9,429,324 gradients, 21.6 GFLOPs

# 加载一个预训练的 YOLO11n 模型
model = YOLO("/data/lijunlin/project/Detect/yolov11/CL/yolo11n-CL.yaml")

# 在 COCO8 数据集上训练模型 100 个周期
train_results = model.train(
    data         = "/data/lijunlin/project/Detect/yolov11/CL/data_stage1.yaml",  # 数据集配置文件路径
    epochs       = 300,  # 训练周期数
    imgsz        = 50,  # 训练图像尺寸
    device       = "cuda:0",  # 运行设备（例如 'cpu', 0, [0,1,2,3]）
    multi_scale  = True,
    close_mosaic = 10,
    amp          = True,
    fraction     = 1,
    profile      = True,
    batch        = 16,
    cutmix       = 0.2,
    mixup        = 0.2,
    mosaic       = 0.3,
    scale        = 1/3,
)

# # 评估模型在验证集上的性能
# metrics = model.val()

# # 对图像执行目标检测
# results = model("path/to/image.jpg")  # 对图像进行预测
# results[0].show()  # 显示结果

# 将模型导出为 ONNX 格式以进行部署
# path = model.export(
#     format="onnx",
#     dynamic=True,
#     imgsz=768,
# )  # 返回导出模型的路径.
