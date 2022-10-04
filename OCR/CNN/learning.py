import tensorflow as tf
from sklearn.metrics import f1_score 
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from keras.models import Sequential,model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
import tensorflow.keras.backend as K

train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1)
path = './' #datasets path

train_generator = train_datagen.flow_from_directory(os.path.join(path, 'train'), target_size=(28,28), class_mode='sparse')

validation_generator = train_datagen.flow_from_directory(os.path.join(path, 'val'), target_size=(28,28), batch_size=1, class_mode='sparse')

test_generator = train_datagen.flow_from_directory(os.path.join(path, 'test'), target_size=(28,28), batch_size=1, class_mode='sparse')

def f1score(y, y_pred):
    return f1_score(y, tf.math.argmax(y_pred, axis=1), average='micro') 

def custom_f1score(y, y_pred):
    return tf.py_function(f1score, (y, y_pred), tf.double)

class stop_training_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_custom_f1score') > 0.99):
              self.model.stop_training = True

def store_keras_model(model, model_name):
    model_json = model.to_json()
    with open("./{}.json".format(model_name), "w") as json_file:
        json_file.write(model_json)
    model.save_weights("./{}.h5".format(model_name))
    print("Saved model to disk")

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

#learning param
batch_size = 10
callbacks = [stop_training_callback()]
epoch = 60
step_save = 10
#learning with save model
for i in range(int(epoch / step_save)):
    model.fit_generator(
          train_generator,
          steps_per_epoch = train_generator.samples // batch_size,
          validation_data = validation_generator, 
          epochs = step_save, verbose=1, callbacks=callbacks)
    store_keras_model(model, 'model_LicensePlate_epoch_' + str((i + 1) * step_save))
