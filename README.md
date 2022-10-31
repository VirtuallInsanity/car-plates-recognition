# Распознавание номеров машин

В файле ***read_plate_img.py*** происходит детекция номера на изображении с помощью yolov5/yolov7/segmentator, а затем считывание символов припомощи CNN/Tesseract с обнаруженных номеров.

В файле ***read_plate_vid.py*** происходит детекция номера на видео с помощью yolov5/yolov7, а затем считывание символов припомощи CNN/Tesseract с обнаруженных номеров.

### Запуск распознавания
1. Используя файл можно установить зависимости requirements.txt:

    <code>python -m pip install -r requirements.txt</code>

2. Скачать обученные модели и сохранить их в соответсвующие папки указав пути к ним в файле **config.ini**.
3. Выбрать модели и указать путь к файлу для распознаванию в файле **config.ini**.
    - Модели для детекции: yolov5, yolov7, segmentator
    - Модели для распознавания символов: cnn, tesseract
4. Запустить ***read_plate_img.py***. Результат выводится в консоль и распознавание сохранятеся в папку **runs**.

# Использованные метрики

Точность — это метрика, которая подсчитывает долю правильных прогнозов. Точность модели обнаружения объектов зависит от качества и количества обучающих выборок, входных изображений, параметров модели и требуемого порога точности.

![2 -Accuracy-formula-machine-learning-algorithms](https://user-images.githubusercontent.com/48131753/199040195-486e91bc-0e5c-4d48-92e3-c256eb2d1bc9.png)

Отношение пересечения к объединению (IoU) используется в качестве порога для определения того, является ли прогнозируемый результат истинно положительным или ложноположительным. Отношение IoU — это степень перекрытия между ограничивающей рамкой вокруг прогнозируемого объекта и ограничивающей рамкой вокруг наземных эталонных данных.

![image](https://user-images.githubusercontent.com/48131753/199042345-cd71fe74-730e-4278-8319-0a28c16c46e9.png)

* еслии IoU ≥ 0,5, классифицируйте обнаружение объекта как True Positive (TP)
* если Iou < 0,5 , то это неправильное обнаружение и классифицировать его как False Positive (FP)


F1-score объединяет в себе информацию о точности (precision) и полноте (recall) модели.

![image](https://user-images.githubusercontent.com/48131753/199041336-5cb91861-6b04-4ee9-90af-1fae55faae81.png)

Оценка F1 представляет собой средневзвешенное значение точности и полноты. Диапазон значений от 0 до 1, где 1 означает наивысшую точность.


# Plates detection
В папке ***car_plates*** находятся следующие файлы:
- ***car_plates/detect_plates_yolov5.py*** содержит метод для использования нейронной сети yolov5 от [ultralytics](https://github.com/ultralytics/yolov5)
- ***car_plates/detect_plates_yolov7.py*** содержит метод для использования нейронной сети yolov7 от [WongKinYiu](https://github.com/WongKinYiu/yolov7)
- ***car_plates/generate_txt.py*** вспомогательный скрипт который был использован для генерации аннотации изображений для сетей yolo в .txt формате
- ***car_plates/yolov5_out/*** содержит веса, инференс на одной картинке и различные метрики обучения относящиеся к yolov5
- ***car_plates/yolov7_out/*** содержит инференс на одной картинке относящийся к yolov7, веса можно скачать по ссылке и поместить в эту папку

Скачать веса, данные для теста и посмотреть исходыне данные: https://drive.google.com/drive/folders/1oxzpehwLPBXs81VrEGzPs4b73ADVO5C3?usp=sharing

Вывод детекции нейронных сетей:
| **yolov7** | **yolov5** |
|----------------|---------|
| ![yolov7](https://github.com/VirtuallInsanity/car-plates-recognition/blob/develop/info/yolov7.jpg) |![car1-resize2](https://github.com/VirtuallInsanity/car-plates-recognition/blob/develop/info/yolov5.jpg) |

Параметры используемых нейронных сетей:
| **parameters** | **yolov7** | **yolov5** |
|----------------|---------|---------|
| **image size** | 640x640 | 640x640 |
| **learning rate initial** | 0.01 | 0.01 |
| **learning rate final** | 0.1 | 0.01 |
| **momentum** | 0.937 | 0.937 |
| **weight_decay** | 0.0005 | 0.0005 |
| **optimizer** | adam | SGD(no decay) |

Вывод быстродействия на одном изображении:
| **yolov7 inference** | **yolov5 inference** |
|----------------|---------|
| 1 carplate, Speed: 1640.3ms, Inference: 0.7ms NMS | 1 carplate, Speed: 0.5ms pre-process, 11.8ms inference, 1.4ms NMS |


# Plates segmentation
### Датасет
Для обучения и валидации модели были использованы данные из этого [соревнования](https://www.kaggle.com/competitions/vkcv2022-contest-02-carplates). Для валидации использовалось 5% изображений.
### Алгоритм работы
1. С помощью семантической сегментации выделяются области изображения, относящиеся к автомобильным номера.
2. Далее из маски вычленяются связанные компоненты. Вокруг них строятся прямоугольники с минимально возможной площадью.
3. На основе координат прямоугольников производится афинное преобразование для «выравнивания» номерного знака. Далее знак может передваться для OCR.
### Предобработка данных
##### Training
```python
import albumentations as al

augmentation = al.Compose
(
    [
        al.PadIfNeeded(
            min_height=config.image_height,
            min_width=config.image_width,
            border_mode=0,
            value=0,
            mask_value=0,
            always_apply=True,
        ),
        al.CropNonEmptyMaskIfExists(
            height=config.image_height,
            width=config.image_width,
        ),
        al.Perspective(
            pad_mode=0,
        ),
        al.OneOf(
            [
                al.Blur(blur_limit=7, p=1.0),
                al.MotionBlur(blur_limit=7, p=1.0),
            ],
            p=0.8,
        ),
    ],
)
```
##### Validation
```python
import albumentations as al

augmentation = al.PadIfNeeded
(
    min_height=None,
    min_width=None,
    pad_height_divisor=config.height_divisor,
    pad_width_divisor=config.width_divisor,
    border_mode=0,
    value=0,
    mask_value=0,
    always_apply=True,
)
```
### Результаты
| Model      | Encoder         | Image size | F1 score validation |
| :--------: | :-------------: | :--------: | :-----------------: |
| Unet       | ResNet-18       | 320x320    | 0.92688             |
| Unet       | ResNet-18       | 480x480    | 0.93687             |
| DeepLabV3+ | ResNet-18       | 320x320    | 0.92314             |
| DeepLabV3+ | ResNet-18       | 480x480    | 0.9243              |
| Unet       | EfficientNet B3 | 480x480    | 0.93004             |
### Пример работы
| ![Segmentation example](https://github.com/VirtuallInsanity/car-plates-recognition/blob/develop/info/example_work.png) | 
|:--------------------------------------------------------:| 
|                *Пример работы алгоритма.*                |
# OCR
Для решения задачи OCR использовался датасет [Nomeroff Russian license plates](https://www.kaggle.com/datasets/evgrafovmaxim/nomeroff-russian-license-plates). Задача OCR была разделена на две подзадачи:
- Выделение символов на номере
- Распознавание символов
Для выделения символов использовался скрипт ***OCR/dataset.py***, который формирует набор символов разделенных по классам по следующему принципу:
- Пороговая фильтрация изображения
- Эрозия и дилатация
- Поиск контуров (по размеру)
- Удаление вложенных контуров
- Обрезание контура и сохранение как отдельного символа

### CNN
В папке ***OCR/CNN*** находятся файлы:
- ***OCR/CNN/learning.py*** для обучения модели
- ***OCR/CNN/use_model.py*** скрипт использования обученной модели на данных

Рассмотрены следующие архитектуры модели:

#### CNN 1
```python
model = Sequential()
model.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (4,4), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dense(22, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=[custom_f1score])
```

Обучение заняло 80 эпох, результирующая Accuracy на тестовых данных составил около 0,95. Модель сохранена в ***OCR/CNN/model_LicensePlate_1.json*** и ***OCR/CNN/model_LicensePlate_1.h5***

#### CNN 2
```python
model = Sequential()
model.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (4,4), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(36, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=[custom_f1score])
```

Обучение заняло также 80 эпох, результирующая Accuracy на тестовых данных составил около 0.987. Модель сохранена в ***OCR/CNN/model_LicensePlate_2.json*** и ***OCR/CNN/model_LicensePlate_2.h5***

### Tesseract
***OCR/Tesseract/use_model.py*** скрипт использования модели на данных

Результирующая Accuracy на тестовых данных составил около 0,6.

### Results
| Модель OCR | Accuracy | Время распознавания одного символа, с|
|:----:|:----:|:----------:|
| CNN 1 | 0.95 | 0.065 |
| CNN 2 | 0.987 | 0.07 |
| Tesseract | 0.4 | 0.14 |

### TESTS

#### CNN2, [dataset](https://www.kaggle.com/datasets/evgrafovmaxim/nomeroff-russian-license-plates)

Итоговая точность (Accuracy) составила 0.582

#### CNN2 + Plates segmentation, [dataset](https://www.kaggle.com/competitions/vkcv2022-contest-02-carplates)

Итоговая точность (Accuracy) составила 0.126

#### Resume

Было отмечено, что качество OCR снижается при снижении качества и освещенности изображений, поступающих на вход. Например, следующие изображения хорошо детекцируются:
![test1](https://github.com/VirtuallInsanity/car-plates-recognition/blob/develop/info/test1.png)
![test2](https://github.com/VirtuallInsanity/car-plates-recognition/blob/develop/info/test2.png)

Примером плоходетектируемых изображений являются:
![test3](https://github.com/VirtuallInsanity/car-plates-recognition/blob/develop/info/test3.png)
![test4](https://github.com/VirtuallInsanity/car-plates-recognition/blob/develop/info/test4.png)

Способы увеличить точность:
- Тесты с настройкой параметров поиска символов на номере
- Решение задачи с изображениями высокого качества
- Адаптация параметров поиска символов под освещенность изображения

Программное решение лучше всего подойдет для следующих задач:
- Детекция правонарушений фиксированной камерой над перекрестком
- Детекция автомобилей на КПП/парковках
- Прочие подобные задачи

### Conclusion
Лучше всего для детекции себя показала yolov7. Имея такую же, а в некоторых кейсах(плохая освещенность и т.д.) и более лучшую точность по сравнению с сегментатором. Так же, быстрота работы семейства yolo с видео, является весомым преимуществом перед сегментатором. Однако установка и отсутствие нативной поддержки PyTorch осложняет работу с yolov7, что может выставить yolov5 в более выгодном свете несмотря на более низкую точность.
