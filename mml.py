import random
from tensorflow.keras.models import Sequential # Подлючаем класс создания модели Sequential
from tensorflow.keras.layers import Dense # Подключаем класс Dense - полносвязный слой
from tensorflow.keras.optimizers import Adam # Подключаем оптимизатор Adam
import numpy as np # Подключаем библиотеку numpy

model = Sequential() # Создаём сеть прямого распространения

model.add(Dense(2, input_dim=2, activation="relu",use_bias=False)) # Добавляем полносвязный слой на 800 нейронов с relu-активацией
model.add(Dense(1, activation="relu",use_bias=False)) # Добавляем полносвязный слой на 800 нейронов с relu-активацией
model.compile(optimizer=Adam(0.001), loss='mse') # Альтернативный способ указания оптимизатора

model.load_weights('model.h5')
x1 = random.randint(0,100) # Установим значение x1
x2 = random.randint(0,100) # Установим значение x2

x_train = np.expand_dims(np.array([x1, x2]), 0) # Создадим набор данных для последующего обучения нейронной сети
y_train = np.expand_dims(np.array([x1+x2]), 0)

l = model.train_on_batch(x_train, y_train) # Обучаем модель на одном наборе данных методом train_on_batch. Метод вернет ошибку и метрику

for i in range(1000): # Пройдемся в цикле 1000 раз
  loss = model.train_on_batch(x_train, y_train) # Выполним на каждом шаге обучение нашей модели
print('Ошибка (mse) после 1000 шагов', loss)

print("Значения x1 и x2: ", x1,x2)
y_pred = model.predict(x_train) # Получаем результат нашей модели
print('Значение модели:', round(y_pred[0][0]),0)
