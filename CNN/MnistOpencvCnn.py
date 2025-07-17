import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# 1. Veriyi Yükle
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalize et ve yeniden boyutlandır
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

x_train = np.expand_dims(x_train, -1)  # (num_samples, 28, 28, 1)
x_test  = np.expand_dims(x_test, -1)

# 3. Etiketleri
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

# 4. Modeli Oluştur
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 5. Modeli Derle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 6. Modeli Eğit
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# 7. Test Verisi ile Değerlendirme
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

# 8. OpenCV ile tahmin görselleştirme
for i in range(10):
    img = x_test[i].reshape(28, 28)
    label = np.argmax(y_test[i])
    prediction = np.argmax(model.predict(np.expand_dims(x_test[i], axis=0)))

    # Görüntüyü büyüt
    img_display = cv2.resize(img, (280, 280), interpolation=cv2.INTER_NEAREST)
    img_display = (img_display * 255).astype(np.uint8)

    # Tahmini ve gerçek etiketi yaz
    cv2.putText(img_display, f"Pred: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.putText(img_display, f"True: {label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

    cv2.imshow("MNIST Prediction", img_display)
    cv2.waitKey(1000)  # 1 saniye bekle
cv2.destroyAllWindows()
