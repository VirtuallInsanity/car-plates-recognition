import cv2
from pathlib import Path
import shutil
import configparser
import OCR.CNN.use_model
import OCR.Tesseract.use_model

from car_plates.detect_plates_yolov5 import inference_yolov5
from car_plates.detect_plates_yolov7 import inference_yolov7

dirpath = Path('runs')
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)

config = configparser.ConfigParser()
config.read("config.ini", encoding='utf-8')

filename = config['Input']['filepath']

detection_model = config['Detection']['model']
weights = config['Detection']['weights_path']
if detection_model == 'yolov5':
    cropped_imgs_path = inference_yolov5(filename, weights_path=weights)
elif detection_model == 'yolov7':
    cropped_imgs_path = inference_yolov7(filename, weights_path=weights)
elif detection_model == 'segmentator':
    # TO-DO
    pass
else:
    print("Incorrect detection model! Choose from: yolov5, yolov7, segmentator")
    exit()

for cropped_img in cropped_imgs_path:
    image = cv2.imread(cropped_img)
    image = cv2.resize(image, (400, 90))
    ocr_model = config['OCR']['model']
    if ocr_model == 'cnn':
        char = OCR.CNN.use_model.segment_characters(image)
        print(OCR.CNN.use_model.show_results(char))
    elif ocr_model == 'tesseract':
        char = OCR.Tesseract.use_model.segment_characters(image)
        print(OCR.Tesseract.use_model.show_results(char))
    else:
        print("Incorrect OCR model! Choose from: cnn, tesseract")
