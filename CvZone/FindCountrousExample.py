import cv2
import cvzone
import numpy as np


def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Hata: {image_path} dosyası bulunamadı.")
        exit()
    return img


def process_image(img, canny_threshold1=50, canny_threshold2=150, dilation_kernel_size=5, dilation_iterations=1):
    img_canny = cv2.Canny(img, canny_threshold1, canny_threshold2)

    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    img_dilated = cv2.dilate(img_canny, kernel, iterations=dilation_iterations)

    return img_canny, img_dilated


def find_and_draw_contours(img, img_dilated, min_area=1000, max_area=100000, sort=True, filter_corners=None,
                           color=(255, 0, 0), thickness=2):
    img_contours, contours_found = cvzone.findContours(
        img, img_dilated, minArea=min_area, maxArea=max_area, sort=sort,
        filter=filter_corners, drawCon=True, c=color, ct=color,
        retrType=cv2.RETR_EXTERNAL, approxType=cv2.CHAIN_APPROX_NONE
    )
    return img_contours, contours_found


def main():
    image_path = "shapes.png"
    img_shapes = load_image(image_path)

    img_canny, img_dilated = process_image(img_shapes)

    img_contours_all, _ = find_and_draw_contours(img_shapes, img_dilated, filter_corners=None)

    img_contours_filtered, _ = find_and_draw_contours(img_shapes, img_dilated, filter_corners=[3, 4])

    img_stacked = cvzone.stackImages([img_shapes, img_canny, img_dilated, img_contours_all, img_contours_filtered],
                                     cols=2, scale=0.8)
    cv2.imshow("Stacked Images", img_stacked)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()