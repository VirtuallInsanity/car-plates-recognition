from PIL import Image
import pytesseract
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def find_contours(dimensions, img) :

    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    #Finding suitable contours
    x_cntr_list = []
    target_contours = []
    img_res = []
    cntrs_int = []
    for cntr in cntrs :
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height:
            cntrs_int.append(cntr)
    
    #Deleting nested contours
    cntrs_last = []
    for cntr in cntrs_int:
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        point = (intX, intY)
        cntrs_last.append(cntr)
        for cntrout in cntrs_int:
            if (cv2.boundingRect(cntrout) != cv2.boundingRect(cntr)) and (cv2.pointPolygonTest(cntrout, point, False) >= 0):
                cntrs_last.pop()
                break
    
    for cntr in cntrs_last:     
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

        x_cntr_list.append(intX) 

        char_copy = np.zeros((44,24))
        char = img[intY:intY+intHeight, intX:intX+intWidth]
        char = cv2.resize(char, (20, 40))
        char = cv2.subtract(255, char)

        char_copy[2:42, 2:22] = char
        char_copy[0:2, :] = 0
        char_copy[:, 0:2] = 0
        char_copy[42:44, :] = 0
        char_copy[:, 22:24] = 0

        img_res.append(char_copy) 
        
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])
    img_res = np.array(img_res_copy)

    return img_res

def segment_characters(image) :

    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]
    #cutting edges
    img_binary_lp[0:10,:] = 255
    img_binary_lp[:,0:20] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255
    #allowed character size
    dimensions = [LP_WIDTH/8,
                  LP_WIDTH/2,
                  LP_HEIGHT/15,
                  5*LP_HEIGHT/6]
    
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list

def show_results(char):
    output = []
    for i,ch in enumerate(char):
        cv2.imwrite('./test.png', ch)
        img = cv2.imread('./test.png')
        character = pytesseract.image_to_string(img, config='--psm 13')[0]
        if (character in ('0', 'O')):
            if (i in (0, 4, 5)): #letters
                character = 'O'
            else:
                character = '0'
        output.append(character)
        
    plate_number = ''.join(output)
    
    return plate_number

if len(sys.argv) == 2:    
    raw_path = sys.argv[1] #path
    path = raw_path.replace('\\', '/').replace('"',"") #\ replace /, delete ""
    img = cv2.imread(path) #'./nomeroff-russian-license-plates/autoriaNumberplateOcrRu-2021-09-01/test/img/A001BP54.png' for example
    char = segment_characters(img)
    print(show_results(char))
else:
    print("Не указан путь!")

# #TEST MODEL
# train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1)
# test_generator = train_datagen.flow_from_directory('./test', target_size=(28,28), batch_size=1, class_mode='sparse')
# ok = 0
# all = len(test_generator.filepaths)
# for i in range(len(test_generator.filepaths)):
#     path = test_generator.filepaths[i]
#     key = test_generator.filepaths[i][33:test_generator.filepaths[i].rfind('/')]
#     symbol = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     #symbol = cv2.bitwise_not(symbol)
#     result = pytesseract.image_to_string(symbol, config='--psm 13')[0]
#     if ('class_' + result == key) or (result in ('0', 'O') and key in ('class_0', 'class_O')) or ((result == 'V') and (key == 'class_Y')):
#         ok += 1
# print(ok/all)
