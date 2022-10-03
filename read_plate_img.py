import cv2
from pathlib import Path
import shutil

from car_plates.detect_plates_yolov5 import inference_yolov5
# from OCR.Tesseract.use_model import segment_characters, show_results
from OCR.CNN.use_model import segment_characters, show_results

filename = 'cars1.bmp'
cropped_imgs_path = inference_yolov5(filename)

for cropped_img in cropped_imgs_path:
    char = segment_characters(cv2.imread(cropped_img))
    print(show_results(char))

dirpath = Path('runs')
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)
