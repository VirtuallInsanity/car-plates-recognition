import numpy as np
import cv2
import os
import json

def find_contours(dimensions, img) :

    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = cv2.imread('contour.jpg')
    
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
        cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
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

train_symbols = []
path_json = './nomeroff-russian-license-plates/autoriaNumberplateOcrRu-2021-09-01/train/ann/'
for dirname, _, filenames in os.walk('./nomeroff-russian-license-plates/autoriaNumberplateOcrRu-2021-09-01/train/img'):
    for filename in filenames:
        name = json.load(open(path_json + filename.replace('.png', '.json')))
        train_symbols.append((os.path.join(dirname, filename), name['description']))
#create train dataset
for i in range(len(train_symbols)):
    char = segment_characters(cv2.imread(train_symbols[i][0]))
    if len(char) == (len(train_symbols[i][1])):
        for j in range(len(train_symbols[i][1])):
            if (os.path.exists('./train/class_' + train_symbols[i][1][j]) == False):
                os.mkdir('./train/class_' + train_symbols[i][1][j])
            cv2.imwrite('./train/class_' + train_symbols[i][1][j] + '/' + train_symbols[i][1] + '_' + str(j) + '.jpg', char[j])

val_symbols = []
path_json = './nomeroff-russian-license-plates/autoriaNumberplateOcrRu-2021-09-01/val/ann/'
for dirname, _, filenames in os.walk('./nomeroff-russian-license-plates/autoriaNumberplateOcrRu-2021-09-01/val/img'):
    for filename in filenames:
        name = json.load(open(path_json + filename.replace('.png', '.json')))
        val_symbols.append((os.path.join(dirname, filename), name['description']))
#create val dataset
for i in range(len(val_symbols)):
    char = segment_characters(cv2.imread(val_symbols[i][0]))
    if len(char) == (len(val_symbols[i][1])):
        for j in range(len(val_symbols[i][1])):
            if (os.path.exists('./val/class_' + val_symbols[i][1][j]) == False):
                os.mkdir('./val/class_' + val_symbols[i][1][j])
            cv2.imwrite('./val/class_' + val_symbols[i][1][j] + '/' + val_symbols[i][1] + '_' + str(j) + '.jpg', char[j])

test_symbols = []
path_json = '.nomeroff-russian-license-plates/autoriaNumberplateOcrRu-2021-09-01/test/ann/'
for dirname, _, filenames in os.walk('./nomeroff-russian-license-plates/autoriaNumberplateOcrRu-2021-09-01/test/img'):
    for filename in filenames:
        name = json.load(open(path_json + filename.replace('.png', '.json')))
        test_symbols.append((os.path.join(dirname, filename), name['description']))
#create test dataset
for i in range(len(test_symbols)):
    char = segment_characters(cv2.imread(test_symbols[i][0]))
    if len(char) == (len(test_symbols[i][1])):
        for j in range(len(test_symbols[i][1])):
            if (os.path.exists('./test/class_' + test_symbols[i][1][j]) == False):
                os.mkdir('./test/class_' + test_symbols[i][1][j])
            cv2.imwrite('./test/class_' + test_symbols[i][1][j] + '/' + test_symbols[i][1] + '_' + str(j) + '.jpg', char[j])
