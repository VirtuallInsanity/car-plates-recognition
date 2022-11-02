import time

import cv2
import configparser
import OCR.CNN.use_model
import OCR.Tesseract.use_model

from car_plates.detect_plates_yolov5 import load_model_yolov5
from car_plates.detect_plates_yolov7 import load_model_yolov7

config = configparser.ConfigParser()
config.read("config.ini", encoding='utf-8')

label_font = cv2.FONT_HERSHEY_COMPLEX  # Font for the label
detection_model = config['Detection']['model']
weights = config['Detection']['weights_path']
ocr_model = config['OCR']['model']
filename = config['Input']['filepath']

if detection_model == 'yolov5':
    model = load_model_yolov5(weights_path=weights)
elif detection_model == 'yolov7':
    model = load_model_yolov7(weights_path=weights)
else:
    print("Incorrect detection model! Choose from: yolov5, yolov7")
    exit()

cap = cv2.VideoCapture(filename)

if not cap.isOpened():
    print("Error opening video stream or file")

# Read until video is completed
while cap.isOpened():
    start = time.time()
    ret, frame = cap.read()
    if ret:
        results = model(frame)
        labels = results.xyxyn[0][:, -1].cpu().numpy()
        cord = results.xyxyn[0][:, :-1].cpu().numpy()

        n = len(labels)

        x_shape, y_shape = frame.shape[1], frame.shape[0]

        for i in range(n):
            row = cord[i]
            # If score is less than 0.3 we avoid making a prediction
            if row[4] < 0.3:
                continue
            x1 = int(row[0] * x_shape)
            y1 = int(row[1] * y_shape)
            x2 = int(row[2] * x_shape)
            y2 = int(row[3] * y_shape)
            bgr = (0, 255, 0)  # color of the box
            classes = model.names  # Get the name of label index
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)  # Plot the boxes

            image = cv2.resize(frame[y1:y2, x1:x2], (400, 90))
            if ocr_model == 'cnn':
                char = OCR.CNN.use_model.segment_characters(image)
                out = OCR.CNN.use_model.show_results(char)
                print(out)
            elif ocr_model == 'tesseract':
                char = OCR.Tesseract.use_model.segment_characters(image)
                out = OCR.Tesseract.use_model.show_results(char)
                print(out)
            else:
                out = 'no ocr!'
                print("Incorrect OCR model! Choose from: cnn, tesseract")

            cv2.putText(frame, out, (x1, y1), label_font, 1, bgr, 2)

        fin = time.time()
        fps = str(int(1 / (fin - start)))
        cv2.putText(frame, f'FPS: {fps}', (5, 30),
                    label_font, 1, (100, 255, 0), 3, cv2.LINE_AA)

    img = cv2.resize(frame, (frame.shape[1], frame.shape[0]))
    cv2.imshow('detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
