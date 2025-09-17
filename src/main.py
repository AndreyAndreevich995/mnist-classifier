import os
import logging
from datetime import datetime
import warnings

# Отключение всех предупреждений
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключает все сообщения кроме FATAL
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Отключает oneDNN оптимизации
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Принудительно использует CPU
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir='  # Отключает XLA GPU

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Input
from keras.utils import to_categorical

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Настройка логгера
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs('logs', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Настройка логгера для записи в файл и консоль
log_filename = f'logs/training_log_{current_time}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 1) Данные
logger.info("Начало загрузки данных MNIST")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

logger.info(f"Размер обучающего набора: {x_train.shape}")
logger.info(f"Размер тестового набора: {x_test.shape}")
logger.info(f"Количество классов: {len(np.unique(y_train))}")

# Нормализация
logger.info("Нормализация данных")
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# One-hot encoding
logger.info("Применение One-hot encoding")
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

# 2) Модель
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

# 3) Компиляция
logger.info("Компиляция модели")
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

logger.info("Начало обучения модели")

# 4) Обучение
history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

logger.info("Обучение модели завершено")

# 5) Оценка

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print(f"\nИтоговые результаты:")
logger.info(f"Потери на тестовых данных: {test_loss:.4f}")
logger.info(f"Точность на тестовых данных: {test_acc:.4f} ({test_acc*100:.2f}%)")

# Получаем предсказания для более детального анализа
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Вычисляем точность по классам
from sklearn.metrics import classification_report, confusion_matrix
classification_rep = classification_report(y_true_classes, y_pred_classes)

# 6) Визуализация и сохранение графиков
plt.style.use('default')
plt.figure(figsize=(15, 10))

# График 1: Loss
plt.subplot(2, 2, 1)
plt.plot(history.history["loss"], label="Train loss", linewidth=2)
plt.plot(history.history["val_loss"], label="Val loss", linewidth=2)
plt.title("Функция потерь", fontsize=14, fontweight='bold')
plt.xlabel("Эпоха", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# График 2: Accuracy
plt.subplot(2, 2, 2)
plt.plot(history.history["accuracy"], label="Train acc", linewidth=2)
plt.plot(history.history["val_accuracy"], label="Val acc", linewidth=2)
plt.title("Точность", fontsize=14, fontweight='bold')
plt.xlabel("Эпоха", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# График 3: Матрица ошибок
plt.subplot(2, 2, 3)
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Матрица ошибок', fontsize=14, fontweight='bold')
plt.colorbar()
tick_marks = np.arange(10)
plt.ylabel('Истинные метки', fontsize=12)
plt.xlabel('Предсказанные метки', fontsize=12)

# График 4: Примеры ошибочных предсказаний
plt.subplot(2, 2, 4)
# Найдем несколько неправильно классифицированных примеров
wrong_indices = np.where(y_pred_classes != y_true_classes)[0]
if len(wrong_indices) > 0:
    wrong_idx = wrong_indices[0]
    plt.imshow(x_test[wrong_idx], cmap='gray')
    plt.title(f'Ошибка: истинная={y_true_classes[wrong_idx]}, предсказ={y_pred_classes[wrong_idx]}',
              fontsize=12, fontweight='bold')
    plt.axis('off')
else:
    plt.text(0.5, 0.5, 'Ошибок не найдено!', horizontalalignment='center',
             verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
    plt.axis('off')

plt.tight_layout()

# Сохраняем графики с датой и временем
training_results_filename = f'results/training_results_{current_time}.png'
plt.savefig(training_results_filename, dpi=300, bbox_inches='tight')

# Дополнительно создадим отдельные графики
plt.figure(figsize=(12, 5))

# Отдельный график для Loss
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train loss", linewidth=2, color='red')
plt.plot(history.history["val_loss"], label="Val loss", linewidth=2, color='blue')
plt.title("Функция потерь", fontsize=14, fontweight='bold')
plt.xlabel("Эпоха", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Отдельный график для Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train acc", linewidth=2, color='green')
plt.plot(history.history["val_accuracy"], label="Val acc", linewidth=2, color='orange')
plt.title("Точность", fontsize=14, fontweight='bold')
plt.xlabel("Эпоха", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
loss_accuracy_filename = f'results/loss_and_accuracy_{current_time}.png'
plt.savefig(loss_accuracy_filename, dpi=300, bbox_inches='tight')
