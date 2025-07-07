import os
import cv2
import requests

# List of image URLs showing people walking (pedestrians)
image_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/d/d2/People_walking_crosswalk.jpg",
    "https://images.unsplash.com/photo-1504384308090-c894fdcc538d",
    "https://matadornetwork.com/read/wp-content/uploads/2019/02/people-crossing-street-in-amsterdam.jpg",
    "https://www.thewonderofwandering.com/wp-content/uploads/2020/08/walking-europe.jpg"
]

os.makedirs("/mnt/data/yaya_gorselleri", exist_ok=True)
downloaded_images = []

# Download each image from the URLs and save locally
for idx, url in enumerate(image_urls, start=1):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img_path = f"/mnt/data/yaya_gorselleri/yaya_{idx}.jpg"
            with open(img_path, "wb") as f:
                f.write(response.content)
            downloaded_images.append(img_path)
    except Exception as e:
        continue

def detect_people_in_image(image_path, output_dir="/mnt/data/yaya_gorselleri/detected"):
    image = cv2.imread(image_path)
    if image is None:
        return None

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    (rects, _) = hog.detectMultiScale(image, padding=(8, 8), scale=1.05)

    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, image)

    return output_path

# Detect people in each downloaded image and save results
output_paths = [detect_people_in_image(img_path) for img_path in downloaded_images if detect_people_in_image(img_path)]

# Output list of saved detected images
output_paths
