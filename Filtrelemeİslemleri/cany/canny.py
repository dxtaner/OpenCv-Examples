import cv2
import numpy as np
from matplotlib import pyplot as plt

resim = cv2.imread("messi.jpg",0)
kenarlar = cv2.Canny(resim,50,100)

plt.subplots(121),plt.imshow(resim,cmap="gray")
plt.title("orjinalRes"),plt.xticks([]),plt.yticks([])
plt.subplots(122),plt.imshow(kenarlar,cmap="gray")
plt.title("kenarlarRes"),plt.xticks([]),plt.yticks([])
plt.show()
