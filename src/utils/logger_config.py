"""
Настройка системы логирования
"""
import os
import logging
from datetime import datetime

def setup_logger():
    """Настройка логгера для записи в файл и консоль"""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Создание директорий для логов и результатов
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
    logger.info(f"Логирование настроено. Файл лога: {log_filename}")

    return logger
