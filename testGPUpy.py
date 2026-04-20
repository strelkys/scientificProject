import tensorflow as tf
import time

# Разрешить динамическое выделение памяти
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Память GPU выделяется динамически")
    except RuntimeError as e:
        print(e)

print("=" * 60)
print("🔍 ПОЛНАЯ ДИАГНОСТИКА GPU ДЛЯ TENSORFLOW")
print("=" * 60)

# 1. Основная информация
print(f"\n📦 TensorFlow version: {tf.__version__}")
print(f"🔧 Built with CUDA: {tf.test.is_built_with_cuda()}")

# 2. Проверка GPU
print("\n" + "=" * 60)
print("🎮 ИНФОРМАЦИЯ О GPU")
print("=" * 60)

gpus = tf.config.list_physical_devices('GPU')
print(f"Количество физических GPU: {len(gpus)}")

if gpus:
    for i, gpu in enumerate(gpus):
        print(f"\nGPU {i}:")
        print(f"  Имя: {gpu.name}")
        print(f"  Тип: {gpu.device_type}")

    # Логические устройства
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"\nКоличество логических GPU: {len(logical_gpus)}")

    # 3. Тест производительности
    print("\n" + "=" * 60)
    print("⚡ ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 60)

    # Тест 1: Матричное умножение
    print("\n📊 Тест 1: Матричное умножение (10000x10000)")
    try:
        with tf.device('/GPU:0'):
            start = time.time()
            x = tf.random.normal([10000, 10000])
            y = tf.matmul(x, x)
            elapsed = time.time() - start
        print(f"   Время выполнения: {elapsed:.3f} сек")
        print(f"   ✅ Успешно!")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")

    # Тест 2: Свертка (CNN операция)
    print("\n📊 Тест 2: Свертка (имитация CNN)")
    try:
        with tf.device('/GPU:0'):
            start = time.time()
            input_tensor = tf.random.normal([32, 224, 224, 3])
            kernel = tf.random.normal([3, 3, 3, 64])
            output = tf.nn.conv2d(input_tensor, kernel, strides=1, padding='SAME')
            elapsed = time.time() - start
        print(f"   Время выполнения: {elapsed:.3f} сек")
        print(f"   ✅ Успешно!")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")

    # 4. Проверка памяти GPU
    print("\n" + "=" * 60)
    print("💾 ИНФОРМАЦИЯ О ПАМЯТИ GPU")
    print("=" * 60)

    try:
        for gpu in gpus:
            memory_info = tf.config.experimental.get_memory_info(gpu.name)
            print(f"\n{gpu.name}:")
            print(f"  Текущее использование: {memory_info['current'] / 1024:.0f} MB")
            print(f"  Пиковое использование: {memory_info['peak'] / 1024:.0f} MB")
    except:
        print("  ⚠️ Не удалось получить информацию о памяти")

    # 5. Настройка роста памяти (рекомендация)
    print("\n" + "=" * 60)
    print("⚙️ РЕКОМЕНДАЦИИ")
    print("=" * 60)
    print("""
Для оптимальной работы с GPU добавьте в начало кода:

    # Разрешить динамическое выделение памяти
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Память GPU выделяется динамически")
        except RuntimeError as e:
            print(e)

Это предотвратит захват всей памяти GPU сразу.
    """)

    print("\n" + "=" * 60)
    print("🎉 GPU ГОТОВ К РАБОТЕ!")
    print("=" * 60)

else:
    print("\n❌ GPU НЕ НАЙДЕН!")
    print("\n🔧 Возможные причины:")
    print("  1. Контейнер запущен без флага --gpus all")
    print("  2. Не установлены драйверы NVIDIA")
    print("  3. Видеокарта не поддерживается")
    print("\n💡 Решение:")
    print("  docker run --gpus all -p 8888:8888 tensorflow/tensorflow:2.15.0-gpu-jupyter")

print("=" * 60)