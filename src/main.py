"""
Основной файл для запуска классификатора MNIST
"""
# ВАЖНО: config должен импортироваться первым для подавления предупреждений TensorFlow
from models.config import setup_environment

from utils.logger_config import setup_logger
from models.data_processor import load_and_preprocess_data
from models.neural_network import create_model, train_model, evaluate_model
from utils.visualizer import create_visualizations

def main():
    """Основная функция для запуска обучения и тестирования модели"""

    # Настройка логирования
    logger = setup_logger()
    logger.info("Запуск программы классификации MNIST")

    # Загрузка и предобработка данных
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    # Создание модели
    model = create_model()

    # Обучение модели
    history = train_model(model, x_train, y_train)

    # Оценка модели
    test_loss, test_acc = evaluate_model(model, x_test, y_test)

    # Создание визуализаций
    create_visualizations(history, model, x_test, y_test)

    logger.info("Программа завершена успешно")

if __name__ == "__main__":
    main()
