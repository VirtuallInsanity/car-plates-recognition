# car-plates-recognition
Скачать веса, данные для теста и посмотреть исходыне данные: https://drive.google.com/drive/folders/1oxzpehwLPBXs81VrEGzPs4b73ADVO5C3?usp=sharing

# OCR
Для решения задачи OCR использовался датасет [Nomeroff Russian license plates](https://www.kaggle.com/datasets/evgrafovmaxim/nomeroff-russian-license-plates). Задача OCR была разделена на две подзадачи:
- Выделение символов на номере
- Распознавание символов
Для выделения символов использовался скрипт ***OCR/dataset.py***, который формирует набор символов разделенных по классам.

### CNN
В папке ***OCR/CNN*** находятся файлы:
- ***OCR/CNN/learning.py*** для обучения модели
- ***OCR/CNN/use_model.py*** скрипт использования обученной модели на данных

Архитектура модели:
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

Обучение заняло 80 эпох, результирующий _loss_ на тестовых данных составил около 0,05.

### Tesseract
***OCR/Tesseract/use_model.py*** скрипт использования модели на данных

Результирующий _loss_ на тестовых данных составил около 0,6.
