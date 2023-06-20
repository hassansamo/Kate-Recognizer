import os
import cv2

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

id_input = input('Enter your ID: ')
sample_num = 0

image_dir = "Kate"

for filename in os.listdir(image_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imwrite("dataSet/User." + id_input + '.' +
                        str(sample_num) + ".jpg", gray[y:y+h, x:x+w])
            sample_num += 1
            cv2.imshow('frame', img)
            cv2.waitKey(100)
    else:
        continue

    if sample_num > 60:
        break

cv2.destroyAllWindows()