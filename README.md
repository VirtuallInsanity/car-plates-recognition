# Распознавание номеров машин

В файле ***read_plate_img.py*** происходит детекция номера на изображении с помощью yolov5/yolov7, а затем считывание символов припомощи CNN/Tesseract с обнаруженных номеров.

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
| ![car1-resize](https://user-images.githubusercontent.com/48131753/194024210-031ddfe7-5c06-42bf-bce1-4ab3aa76af8f.jpg) |![car1-resize2](https://user-images.githubusercontent.com/48131753/194024776-c45b288c-1f54-4339-bf57-a689b122acc8.jpg) |

Вывод детекции нейронных сетей:
| **yolov7** | **yolov5** |
|----------------|---------|
| 640x640 | 640x640 |
| lr0=0.01 | lr0=0.01 |
| lrf=0.1 | lrf=0.01 |
| momentum=0.937 | momentum=0.937 |
| weight_decay=0.0005 | weight_decay=0.0005 |
|  **optimizer: adam** | **optimizer: SGD(no decay)** |

Вывод быстродействия на одном изображении:
| **yolov7 inference** | **yolov5 inference** |
|----------------|---------|
| 1 carplate, Speed: 1640.3ms, Inference: 0.7ms NMS | 1 carplate, Speed: 0.5ms pre-process, 11.8ms inference, 1.4ms NMS |


# Plates segmentation
### Датасет
Для обучения и валидации модели были использованы данные из этого [соревнования](https://www.kaggle.com/competitions/vkcv2022-contest-02-carplates). Для валидации использовалось 5% изображений.
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
| Model      | Image size | F1 score validation |
| :--------: | :--------: | :-----------------: |
| Unet       | 320x320    | 0.92688             |
| Unet       | 480x480    | 0.93687             |
| DeepLabV3+ | 320x320    | in progress...      |
| DeepLabV3+ | 480x480    | in progress...      |

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

Обучение заняло 80 эпох, результирующий _loss_ на тестовых данных составил около 0,05. Модель сохранена в ***OCR/CNN/model_LicensePlate_1.json*** и ***OCR/CNN/model_LicensePlate_1.h5***

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

Обучение заняло также 80 эпох, результирующий _loss_ на тестовых данных составил около 0,013. Модель сохранена в ***OCR/CNN/model_LicensePlate_2.json*** и ***OCR/CNN/model_LicensePlate_2.h5***

### Tesseract
***OCR/Tesseract/use_model.py*** скрипт использования модели на данных

Результирующий _loss_ на тестовых данных составил около 0,6.

### Results
| Модель OCR | Loss на тесте | Время распознавания одного символа, с|
|:----:|:----:|:----------:|
| CNN 1 | 0.05 | 0.065 |
| CNN 2 | 0.013 | 0.07 |
| Tesseract | 0.6 | 0.14 |

### TESTS

#### CNN2, [dataset](https://www.kaggle.com/datasets/evgrafovmaxim/nomeroff-russian-license-plates)

Итоговая точность составила 0.582

#### CNN2 + Plates segmentation, [dataset](https://www.kaggle.com/competitions/vkcv2022-contest-02-carplates)

Итоговая точность составила 0.126

#### Resume

Было отмечено, что качество OCR снижается при снижении качества и освещенности изображений, поступающих на вход. Например, следующие изображения хорошо детекцируются:
![image](https://user-images.githubusercontent.com/70758674/193918875-7e97eaf2-a436-4907-80d3-d2ed5f4634b9.png)
![image](https://user-images.githubusercontent.com/70758674/193918944-ef5a08da-8edf-4972-97fa-b7226df53729.png)

Примером плоходетектируемых изображений являются:
![image](https://user-images.githubusercontent.com/70758674/193919052-de552385-e266-441a-9850-83b937588e1a.png)
![image](https://user-images.githubusercontent.com/70758674/193919075-8632d066-4aab-4afa-84ae-b736c01e91c1.png)

Способы увеличить точность:
- Тесты с настройкой параметров поиска символов на номере
- Решение задачи с изображениями высокого качества
- Адаптация параметров поиска символов под освещенность изображения

Программное решение лучше всего подойдет для следующих задач:
- Детекция правонарушений фиксированной камерой над перекрестком
- Детекция автомобилей на КПП/парковках
- Прочие подобные задачи
