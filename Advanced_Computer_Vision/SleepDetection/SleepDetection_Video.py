import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

# Video akışını aç
cap = cv2.VideoCapture("video1.mp4")

# FaceMesh dedektörünü başlat
detector = FaceMeshDetector()

# Görselleştirmek için LivePlot nesnesi oluştur
plotY = LivePlot(540, 360, [10, 60])

# Yüzdeki belirli noktalar için landmark ID'lerini tanımla
id_list = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]

# Göz kırpma oranlarını depolamak için boş bir liste oluştur
ratio_list = []

# Yüz üzerine daireler çizmek için bir renk tanımla
color = (0, 0, 255)

# Göz kırpma ve renk değişimini izlemek için sayaçları başlat
counter = 0
blink_counter = 0

while True:
    success, img = cap.read()

    if not success:
        break

    # Karedeki yüzü ve yüz hatlarını tespit et
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]

        # Yüzde belirtilen noktalara daireler çiz
        for id in id_list:
            cv2.circle(img, face[id], 5, color, cv2.FILLED)

        # Göz kırpma oranını hesaplamak için belirli noktaları tanımla
        left_up = face[159]
        left_down = face[23]
        left_left = face[130]
        left_right = face[243]

        # Noktalar arasındaki dikey ve yatay uzaklıkları hesapla
        length_ver, _ = detector.findDistance(left_up, left_down)
        length_hor, _ = detector.findDistance(left_left, left_right)

        # Hesaplanan uzaklıkları görselleştirmek için çizgiler çiz
        cv2.line(img, left_up, left_down, (0, 255, 0), 3)
        cv2.line(img, left_left, left_right, (0, 255, 0), 3)

        # Göz kırpma oranını hesapla
        ratio = int((length_ver / length_hor) * 100)
        ratio_list.append(ratio)

        # Ortalama hesaplaması için son 3 göz kırpma oranını sakla
        if len(ratio_list) > 3:
            ratio_list.pop(0)

        ratio_avg = sum(ratio_list) / len(ratio_list)
        print(f"Average Blink Ratio: {ratio_avg:.2f}")

        # Bir göz kırpma tespit edilip edilmediğini belirle ve rengi değiştir
        if ratio_avg < 35 and counter == 0:
            blink_counter += 1
            color = (0, 255, 0)
            counter = 1

        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0
                color = (0, 0, 255)

        # Göz kırpma sayısını resim üzerine yazdır
        cvzone.putTextRect(img, f"Blink Count: {blink_counter}", (50, 50), colorR=color)

        # Grafiği güncelle ve görselleştirilmiş grafik görüntüsünü al
        img_plot = plotY.update(ratio_avg, color)
        img = cv2.resize(img, (640, 360))
        img_stack = cvzone.stackImages([img, img_plot], 2, 1)

        cv2.imshow("video", img_stack)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Video akışını serbest bırak
cap.release()
cv2.destroyAllWindows()
