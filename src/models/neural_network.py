"""
Модель нейронной сети для классификации MNIST
"""
# Импортируем config первым для подавления предупреждений
from . import config
import logging
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Input

def create_model():
    """Создание и компиляция модели нейронной сети"""
    logger = logging.getLogger(__name__)

    logger.info("Создание модели")
    model = Sequential([
        Input(shape=(28, 28)),             # входной слой
        Flatten(),                         # разворачиваем картинку в вектор
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.1),
        Dense(10, activation="softmax")    # выходной слой для классификации
    ])

    print("\nАрхитектура модели:")
    model.summary()

    # Компиляция
    logger.info("Компиляция модели")
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def train_model(model, x_train, y_train):
    """Обучение модели"""
    logger = logging.getLogger(__name__)

    logger.info("Начало обучения модели")

    history = model.fit(
        x_train, y_train,
        epochs=15,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    logger.info("Обучение модели завершено")
    return history

def evaluate_model(model, x_test, y_test):
    """Оценка модели на тестовых данных"""
    logger = logging.getLogger(__name__)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"\nИтоговые результаты:")
    logger.info(f"Потери на тестовых данных: {test_loss:.4f}")
    logger.info(f"Точность на тестовых данных: {test_acc:.4f} ({test_acc*100:.2f}%)")

    return test_loss, test_acc
