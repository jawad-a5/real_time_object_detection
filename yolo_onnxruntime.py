import cv2
import numpy as np
import onnxruntime as ort

# COCO class names
classes = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
    "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize image to a 640x640 with unchanged aspect ratio using padding."""
    shape = img.shape[:2]  # current shape (height, width)
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(shape[1] * ratio), int(shape[0] * ratio))  # width, height
    dw = new_shape[1] - new_unpad[0]  # width padding
    dh = new_shape[0] - new_unpad[1]  # height padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    # resize image
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    # add border
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, dw, dh

def preprocess(image):
    img, ratio, dw, dh = letterbox(image, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)   # add batch dimension
    return img, ratio, dw, dh

def xywh2xyxy(box):
    """Convert [x_center, y_center, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = box
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return [x1, y1, x2, y2]

# Load ONNX model with ONNX Runtime
session = ort.InferenceSession("yolov5s.onnx")
input_name = session.get_inputs()[0].name

# Start webcam
cap = cv2.VideoCapture(0)

conf_threshold = 0.25
iou_threshold = 0.45

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    input_tensor, ratio, dw, dh = preprocess(frame)

    # Run inference
    outputs = session.run(None, {input_name: input_tensor})

    preds = outputs[0]  # shape: (1, num_preds, 85)

    boxes = []
    confidences = []
    class_ids = []

    for pred in preds[0]:
        conf = pred[4]
        if conf > conf_threshold:
            scores = pred[5:]
            class_id = np.argmax(scores)
            if scores[class_id] > conf_threshold:
                box = xywh2xyxy(pred[:4])

                # Correct box for padding and resize to original image scale
                x1 = int((box[0] - dw) / ratio)
                y1 = int((box[1] - dh) / ratio)
                x2 = int((box[2] - dw) / ratio)
                y2 = int((box[3] - dh) / ratio)

                # Clip coordinates
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))

                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(float(conf))
                class_ids.append(class_id)

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)

    if indices is not None:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv5 ONNX with ONNX Runtime", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
