import cv2

def kaydet(path,image,kalite=None,compress=None):

    if kalite:
        cv2.imwrite((path,image,[int(cv2.IMWRITE_JPEG_QUALITY),kalite]))
    elif compress:
        cv2.imwrite((path,image[int(cv2.IMWRITE_PNG_COMPRESSION),kalite]))
    else:
        cv2.imwrite(path,image)


def main():
    impath="cicek.jpg"
    img=cv2.imread(impath)

    cv2.imshow("resim",img)

    cikisjpg="cicek2jpg.jpg"
    kaydet(cikisjpg,img,kalite=85)

    ciksipng = "cicek2png.png"
    kaydet(ciksipng, img, compress=3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()