import os
import numpy as np
import cv2
import random


def generate_dataset_with_shapes_and_yolo(
    save_dir,
    image_size=(50, 50),
    positive_ratio=0.8,
    use_noise_background=True,
    total_samples=50,
    shape_type="ring",      # "ring" | "triangle" | "rectangle"
    thickness=3,             # -1: fill, >=1: outline
    dtype = "train"
):
    """
    生成指定形状的数据集 + YOLOv11 标签
    thickness 统一控制填充 / 轮廓
    """
    class_name = ["ring", "triangle", "rectangle"]
    assert shape_type in class_name
    class_id = class_name.index(shape_type)
    img_dir = os.path.join(save_dir, f"images/{dtype}")
    lbl_dir = os.path.join(save_dir, f"labels/{dtype}")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    H, W = image_size
    num_pos = int(total_samples * positive_ratio)
    num_neg = total_samples - num_pos

    def create_background():
        if use_noise_background:
            return np.random.randint(0, 256, (H, W), dtype=np.uint8)
        else:
            return np.zeros((H, W), dtype=np.uint8)

    # ---------------- draw functions ----------------

    def draw_ring(img):
        max_r = min(H, W) // 2 - max(1, abs(thickness)) - 1
        min_r = max(5, abs(thickness) + 2)
        r = random.randint(min_r, max_r)

        cx = random.randint(r + abs(thickness), W - r - abs(thickness))
        cy = random.randint(r + abs(thickness), H - r - abs(thickness))

        cv2.circle(img, (cx, cy), r, 255, thickness)

        outer = r if thickness == -1 else r + thickness
        xmin = max(0, cx - outer)
        ymin = max(0, cy - outer)
        xmax = min(W - 1, cx + outer)
        ymax = min(H - 1, cy + outer)

        return img, xmin, ymin, xmax, ymax

    def draw_rectangle(img):
        w = random.randint(12, W // 2)
        h = random.randint(12, H // 2)

        xmin = random.randint(0, W - w - 1)
        ymin = random.randint(0, H - h - 1)
        xmax = xmin + w
        ymax = ymin + h

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), 255, thickness)

        return img, xmin, ymin, xmax, ymax

    def draw_triangle(img):
        size = random.randint(12, min(H, W) // 2)

        x0 = random.randint(0, W - size - 1)
        y0 = random.randint(0, H - size - 1)

        pts = np.array([
            [x0, y0],
            [x0 + size, y0],
            [x0, y0 + size]
        ], np.int32)

        if thickness == -1:
            cv2.fillPoly(img, [pts], 255)
        else:
            cv2.polylines(img, [pts], True, 255, thickness)

        xmin, ymin = x0, y0
        xmax, ymax = x0 + size, y0 + size

        return img, xmin, ymin, xmax, ymax

    def draw_shape(img):
        if shape_type == "ring":
            return draw_ring(img)
        elif shape_type == "rectangle":
            return draw_rectangle(img)
        else:
            return draw_triangle(img)

    # ---------------- generation ----------------

    sample_id = 0

    # 正样本
    for _ in range(num_pos):
        img = create_background()
        img, xmin, ymin, xmax, ymax = draw_shape(img)

        # YOLO normalized bbox
        x_center = ((xmin + xmax) / 2) / W
        y_center = ((ymin + ymax) / 2) / H
        bw = (xmax - xmin) / W
        bh = (ymax - ymin) / H

        img_name = f"{shape_type}_{sample_id:04d}.png"
        lbl_name = f"{shape_type}_{sample_id:04d}.txt"

        cv2.imwrite(os.path.join(img_dir, img_name), img)

        with open(os.path.join(lbl_dir, lbl_name), "w") as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

        sample_id += 1

    # 负样本
    for _ in range(num_neg):
        img = create_background()

        img_name = f"{shape_type}_{sample_id:04d}.png"
        lbl_name = f"{shape_type}_{sample_id:04d}.txt"

        cv2.imwrite(os.path.join(img_dir, img_name), img)
        open(os.path.join(lbl_dir, lbl_name), "w").close()

        sample_id += 1

    print(
        f"Generated dataset | shape={shape_type} | thickness={thickness} | "
        f"pos={num_pos}, neg={num_neg}"
    )


if __name__ == "__main__":

    # stage 1
    generate_dataset_with_shapes_and_yolo(
        save_dir="/data/lijunlin/project/Detect/yolov11/CL/data/stage1/",
        image_size=(50, 50),
        positive_ratio=0.8,
        use_noise_background=True,
        total_samples=50,
        shape_type="ring",
        thickness=3,
    )
    
    generate_dataset_with_shapes_and_yolo(
        save_dir="/data/lijunlin/project/Detect/yolov11/CL/data/stage1/",
        image_size=(50, 50),
        positive_ratio=0.8,
        use_noise_background=True,
        total_samples=10,
        shape_type="ring",
        thickness=3,
        dtype="val",
    )

    # stage 2

    generate_dataset_with_shapes_and_yolo(
        save_dir="/data/lijunlin/project/Detect/yolov11/CL/data/stage2/",
        image_size=(50, 50),
        positive_ratio=0.8,
        use_noise_background=True,
        total_samples=10,
        shape_type="ring",
        thickness=3,
    )

    generate_dataset_with_shapes_and_yolo(
        save_dir="/data/lijunlin/project/Detect/yolov11/CL/data/stage2/",
        image_size=(50, 50),
        positive_ratio=0.8,
        use_noise_background=True,
        total_samples=2,
        shape_type="ring",
        thickness=3,
        dtype="val",
    )

    generate_dataset_with_shapes_and_yolo(
        save_dir="/data/lijunlin/project/Detect/yolov11/CL/data/stage2/",
        image_size=(50, 50),
        positive_ratio=1,
        use_noise_background=True,
        total_samples=10,
        shape_type="triangle",
        thickness=3
    )

    generate_dataset_with_shapes_and_yolo(
        save_dir="/data/lijunlin/project/Detect/yolov11/CL/data/stage2/",
        image_size=(50, 50),
        positive_ratio=1,
        use_noise_background=True,
        total_samples=2,
        shape_type="triangle",
        dtype="val",
        thickness=3
    )

   

    # stage 3
    generate_dataset_with_shapes_and_yolo(
        save_dir="/data/lijunlin/project/Detect/yolov11/CL/data/stage3/",
        image_size=(50, 50),
        positive_ratio=0.8,
        use_noise_background=True,
        total_samples=10,
        shape_type="ring",
        thickness=3,
    )

    generate_dataset_with_shapes_and_yolo(
        save_dir="/data/lijunlin/project/Detect/yolov11/CL/data/stage3/",
        image_size=(50, 50),
        positive_ratio=0.8,
        use_noise_background=True,
        total_samples=2,
        shape_type="ring",
        thickness=3,
        dtype="val",
    )

    generate_dataset_with_shapes_and_yolo(
        save_dir="/data/lijunlin/project/Detect/yolov11/CL/data/stage3/",
        image_size=(50, 50),
        positive_ratio=1,
        use_noise_background=True,
        total_samples=10,
        shape_type="triangle",
        thickness=3
    )

    generate_dataset_with_shapes_and_yolo(
        save_dir="/data/lijunlin/project/Detect/yolov11/CL/data/stage3/",
        image_size=(50, 50),
        positive_ratio=1,
        use_noise_background=True,
        total_samples=2,
        shape_type="triangle",
        dtype="val",
        thickness=3
    )

    generate_dataset_with_shapes_and_yolo(
        save_dir="/data/lijunlin/project/Detect/yolov11/CL/data/stage3/",
        image_size=(50, 50),
        positive_ratio=1,
        use_noise_background=True,
        total_samples=10,
        shape_type="rectangle",
        thickness=3
    )

    generate_dataset_with_shapes_and_yolo(
        save_dir="/data/lijunlin/project/Detect/yolov11/CL/data/stage3/",
        image_size=(50, 50),
        positive_ratio=1,
        use_noise_background=True,
        total_samples=2,
        shape_type="rectangle",
        dtype="val",
        thickness=3
    )
