import sys
sys.path.append("..")

from ultralytics import YOLO
# s 181 layers, 9,429,340 parameters, 9,429,324 gradients, 21.6 GFLOPs
import time

start_time = time.time()
def on_train_start(trainer):
    global start_time 
    start_time = time.time() 
 
def on_train_end(trainer):
    total_time = time.time() - start_time
    print(f"Training completed in {total_time}, {total_time / 60:.2f} minutes")

# 加载一个预训练的 YOLO11n 模型
model = YOLO("/data/lijunlin/project/Detect/yolov11/CL/runs/detect/train2/weights/best.pt")
for m in model.model.model[:-13]:
    for p in m.parameters():
        p.requires_grad = False

model.add_callback("on_train_start", on_train_end)  # 注册回调
model.add_callback("on_train_end", on_train_end)  # 注册回调

train_results = model.train(
    data         = "/data/lijunlin/project/Detect/yolov11/CL/data_stage3.yaml",  # 数据集配置文件路径
    epochs       = 70,
    imgsz        = 64,
    device       = "cuda:0",
    batch        = 16,
    lr0          = 1e-1,          
    close_mosaic = 0,
    amp          = True,
    mosaic       = 0.0,         
    mixup        = 0.0,
    cutmix       = 0.0,
)
print(model.info())
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
