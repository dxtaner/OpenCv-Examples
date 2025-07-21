import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Faster RCNN modelini TensorFlow Hub'dan yükle
detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1")

# Görüntü oku
image_path = "Black_square.jpg"
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_tensor = tf.convert_to_tensor(img_rgb)
input_tensor = input_tensor[tf.newaxis, ...]

# Nesne tahmini yap
detections = detector(input_tensor)

# Sonuçları işle
boxes = detections['detection_boxes'][0].numpy()  # Kutu koordinatları [ymin, xmin, ymax, xmax]
scores = detections['detection_scores'][0].numpy()
classes = detections['detection_classes'][0].numpy().astype(np.int32)

height, width, _ = img.shape

# Yüksek güvenlik skoruna sahip kutuları çiz
for i in range(boxes.shape[0]):
    if scores[i] < 0.5:
        continue
    ymin, xmin, ymax, xmax = boxes[i]
    (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
    cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0,255,0), 2)
    label = f"Class {classes[i]}: {scores[i]:.2f}"
    cv2.putText(img, label, (int(left), int(top)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
