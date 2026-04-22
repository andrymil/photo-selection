import cv2
import numpy as np


def get_laplacian_score(img_cv2):
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def get_largest_bbox(yolo_results, image_shape):
    if len(yolo_results.boxes) == 0:
        return None

    boxes = yolo_results.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    largest_box_idx = areas.argmax()
    x1, y1, x2, y2 = map(int, boxes[largest_box_idx])

    h, w = image_shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    return x1, y1, x2, y2


def get_smart_bbox(
    yolo_results, image_shape, weights=(0.3, 0.4, 0.1, 0.2), padding_ratio=0.25
):
    if len(yolo_results.boxes) == 0:
        return None

    boxes = yolo_results.boxes.xyxy.cpu().numpy()
    confidences = yolo_results.boxes.conf.cpu().numpy()
    classes = yolo_results.boxes.cls.cpu().numpy()

    h, w = image_shape[:2]
    img_center_x, img_center_y = w / 2, h / 2
    max_area = h * w
    max_dist = np.sqrt(img_center_x**2 + img_center_y**2)

    best_score = -1
    best_box_idx = -1

    w_area, w_center, w_conf, w_person = weights

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box

        area = (x2 - x1) * (y2 - y1)
        norm_area = area / max_area

        box_center_x = x1 + (x2 - x1) / 2
        box_center_y = y1 + (y2 - y1) / 2
        dist_to_center = np.sqrt(
            (box_center_x - img_center_x) ** 2 + (box_center_y - img_center_y) ** 2
        )
        center_score = 1.0 - (dist_to_center / max_dist)

        conf_score = confidences[i]

        person_score = 1.0 if int(classes[i]) == 0 else 0.0

        score = (
            (w_area * norm_area)
            + (w_center * center_score)
            + (w_conf * conf_score)
            + (w_person * person_score)
        )

        if score > best_score:
            best_score = score
            best_box_idx = i

    x1, y1, x2, y2 = boxes[best_box_idx]

    box_width = x2 - x1
    box_height = y2 - y1

    pad_x = box_width * padding_ratio
    pad_y = box_height * padding_ratio

    x1 = int(x1 - pad_x)
    y1 = int(y1 - pad_y)
    x2 = int(x2 + pad_x)
    y2 = int(y2 + pad_y)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    return x1, y1, x2, y2


# def get_smart_bbox(yolo_results, image_shape, weights=(0.3, 0.4, 0.1, 0.2)):
#     if len(yolo_results.boxes) == 0:
#         return None

#     boxes = yolo_results.boxes.xyxy.cpu().numpy()
#     confidences = yolo_results.boxes.conf.cpu().numpy()
#     classes = yolo_results.boxes.cls.cpu().numpy()

#     h, w = image_shape[:2]
#     img_center_x, img_center_y = w / 2, h / 2
#     max_area = h * w
#     max_dist = np.sqrt(img_center_x**2 + img_center_y**2)

#     best_score = -1
#     best_box_idx = -1

#     w_area, w_center, w_conf, w_person = weights

#     for i, box in enumerate(boxes):
#         x1, y1, x2, y2 = box

#         area = (x2 - x1) * (y2 - y1)
#         norm_area = area / max_area

#         box_center_x = x1 + (x2 - x1) / 2
#         box_center_y = y1 + (y2 - y1) / 2
#         dist_to_center = np.sqrt(
#             (box_center_x - img_center_x) ** 2 + (box_center_y - img_center_y) ** 2
#         )
#         center_score = 1.0 - (dist_to_center / max_dist)

#         conf_score = confidences[i]

#         person_score = 1.0 if int(classes[i]) == 0 else 0.0

#         score = (
#             (w_area * norm_area)
#             + (w_center * center_score)
#             + (w_conf * conf_score)
#             + (w_person * person_score)
#         )

#         if score > best_score:
#             best_score = score
#             best_box_idx = i

#     x1, y1, x2, y2 = map(int, boxes[best_box_idx])
#     x1, y1 = max(0, x1), max(0, y1)
#     x2, y2 = min(w, x2), min(h, y2)

#     return x1, y1, x2, y2
