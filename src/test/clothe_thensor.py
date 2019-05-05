from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense
from keras import utils


# в x_train изображения для обучения
# в y_train правельные ответы

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#60000 изображений по 28*28=784 одномерный пикселей
x_train = x_train.reshape(60000, 784)


#делим на интенсивность пикселя
#x_train /= 255

#преобразуем правильные ответы в категории
y_train = utils.to_categorical(y_train, 10)

#последовательная модель
model = Sequential()

#добавляем уровни сети
#input_dim = 784 - количество входов в нейрон, по количеству
#пикселей в изображении
#relu - Rectified Linear Unit
#softmax - нормальзованная экспоненциальная функция
#сумма всех выходов нейронов равна 1 трактовать как вероятность
model.add(Dense(800, input_dim=784, activation="relu"))
#10 нейронов в выходном слое по количеству классов
model.add(Dense(10, activation="softmax"))

#компилируем модель
#loss - функция ошибки кросжнтропия - хорошо подходит для
#зада классификации если классов больше чем 2
#SGD- стохастический градиентный спуск
#accuracy метрика качества обучения нейронной сети
#доля правильных ответов нейронной сети

model.compile(loss="categorical_crossentropy",
              optimizer="SGD", metrics=["accuracy"])

print(model.summary())

#обучаем сеть
#обучение с учителем
#есть правильные ответы
#градиентный спуск
#размер выборки batch_size=200 , берем 200 изображений рассчитываем
#рассчитываем градиент и изменяем веса
#100 раз обучаем нейронную сеть на одном наборе изображений
#verbose печатаем прогресс обучения нейронной сети
model.fit(x_train, y_train, batch_size=100,
          epochs=2, verbose=1)

# use 60000 images
#loss: 14.5003 - значение фенции ошибки
#acc: 0.1003 - доля правельных ответов в нейронной сети .правильно распознали 10 процентов изображений

#обучение завершено
#запускаем сеть на входных данных
#нужно распознать изображения на которых нейросеть обучалась

predictions = model.predict(x_train)
print(predictions[0])

#выводим номер класса предсказанный нейросетью
#np.argmax(predictions[0])
#библиотека numpy argmax максимальное значение в маccиве

#выводим правильный номер класса из y_train
# print(np.argmax(y_train[0])