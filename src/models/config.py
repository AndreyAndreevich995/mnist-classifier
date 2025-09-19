"""
Конфигурация окружения и подавление предупреждений
Этот модуль должен импортироваться первым, до любого импорта TensorFlow/Keras
"""
import os
import warnings

def setup_environment():
    """Настройка окружения для подавления предупреждений и сообщений"""
    # Отключение всех предупреждений
    warnings.filterwarnings('ignore')

    # Настройки для TensorFlow - должны быть установлены ДО импорта TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключает все сообщения кроме FATAL
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Отключает oneDNN оптимизации
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Принудительно использует CPU
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir='  # Отключает XLA GPU

# Применяем настройки сразу при импорте этого модуля
setup_environment()

# Теперь можно безопасно импортировать TensorFlow
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
