"""
Визуализация результатов обучения и анализ модели
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix

def create_visualizations(history, model, x_test, y_test):
    """Создание и сохранение графиков результатов обучения"""

    # Получаем предсказания для анализа
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Создаем фигуру с графиками
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
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Матрица ошибок', fontsize=14, fontweight='bold')
    plt.colorbar()
    plt.ylabel('Истинные метки', fontsize=12)
    plt.xlabel('Предсказанные метки', fontsize=12)

    # График 4: Примеры неправильных предсказаний
    plt.subplot(2, 2, 4)
    wrong_indices = np.where(y_pred_classes != y_true_classes)[0]
    if len(wrong_indices) > 0:
        wrong_idx = wrong_indices[0]
        plt.imshow(x_test[wrong_idx], cmap='gray')
        plt.title(f'Ошибка: Истинная метка {y_true_classes[wrong_idx]}, '
                 f'Предсказание {y_pred_classes[wrong_idx]}', fontsize=12)
        plt.axis('off')
    else:
        plt.text(0.5, 0.5, 'Нет ошибок!', ha='center', va='center', fontsize=16)
        plt.axis('off')

    plt.tight_layout()

    # Сохранение графика
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'results/training_results_{current_time}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nГрафики сохранены в файл: {filename}")

    plt.show()

    # Вывод детального отчета
    classification_rep = classification_report(y_true_classes, y_pred_classes)
    print(f"\nДетальный отчет по классификации:")
    print(classification_rep)

    return filename
