import cv2


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
