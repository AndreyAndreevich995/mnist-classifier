"""
Загрузка и предобработка данных MNIST
"""
# Импортируем config первым для подавления предупреждений
from . import config
import logging
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

def load_and_preprocess_data():
    """Загрузка и предобработка данных MNIST"""
    logger = logging.getLogger(__name__)

    # Загрузка данных
    logger.info("Начало загрузки данных MNIST")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    logger.info(f"Размер обучающего набора: {x_train.shape}")
    logger.info(f"Размер тестового набора: {x_test.shape}")
    logger.info(f"Количество классов: {len(np.unique(y_train))}")

    # Нормализация
    logger.info("Нормализация данных")
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # One-hot encoding
    logger.info("Применение One-hot encoding")
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)
