# evaluate.py
"""
МОДУЛЬ: evaluate.py
===================
Призначення:
------------
- Завантажує натреновану модель
- Оцінює її точність на тестових даних
- Вимірює середній час інференсу (latency)
"""

import time                              # Для вимірювання часу інференсу
import torch                             # PyTorch для роботи з моделлю і тензорами
import numpy as np                       # NumPy для роботи з масивами та обчисленням середнього
from data_loader import load_data        # Імпорт функції для завантаження test_loader
from model import SpeechCommandCNN       # Імпорт архітектури моделі


# === 1. Функція для обчислення точності ===
def calculate_accuracy(model, test_loader, device):
    model.eval()                         # Переводимо модель у режим оцінки
    correct, total = 0, 0                # Лічильники правильних та всіх відповідей

    with torch.no_grad():                # Вимикаємо обчислення градієнтів
        for inputs, labels in test_loader:             # Проходимо всі батчі тестових даних
            inputs = inputs.to(device)                 # Переносимо тензори на CPU/GPU
            labels = torch.tensor([["yes", "no", "up", "down"].index(l)
                                    for l in labels]).to(device)  # Перетворення текстових міток у індекси

            outputs = model(inputs)                    # Отримуємо прогноз моделі
            _, predicted = torch.max(outputs.data, 1)  # Обираємо клас з найбільшим значенням

            total += labels.size(0)                    # Додаємо кількість прикладів у батчі
            correct += (predicted == labels).sum().item()  # Рахуємо правильні передбачення

    accuracy = 100 * correct / total                   # Обчислюємо точність у %
    return accuracy


# === 2. Функція для вимірювання затримки (latency) ===
def measure_latency(model, test_loader, device, num_batches=10):
    model.eval()                                       # Режим оцінки
    latencies = []                                     # Масив для збереження часу кожного батчу

    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):  # Перебір батчів
            if i >= num_batches:                       # Обмежуємо кількість батчів
                break

            inputs = inputs.to(device)
            start = time.time()                        # Початок вимірювання
            _ = model(inputs)                          # Прогін моделі
            end = time.time()                          # Кінець вимірювання

            latency = (end - start) / len(inputs) * 1000  # Затримка для 1 прикладу в мілісекундах
            latencies.append(latency)                  # Зберігаємо результат

    avg_latency = np.mean(latencies)                   # Середнє значення затримки
    return avg_latency


# === 3. Основна функція оцінки ===
def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Вибираємо CPU або GPU
    print(f"Використовується пристрій: {device}")

    # Завантаження даних
    _, test_loader = load_data(batch_size=32)          # Беремо лише тестовий набір

    # Завантаження моделі
    model = SpeechCommandCNN(num_classes=4).to(device) # Створюємо модель
    model.load_state_dict(torch.load("saved_model/model.pth", map_location=device))  # Завантажуємо ваги
    print("✅ Модель завантажена з saved_model/model.pth")

    # Оцінка точності
    accuracy = calculate_accuracy(model, test_loader, device)
    print(f"🎯 Точність моделі: {accuracy:.2f}%")

    # Вимірювання latency
    avg_latency = measure_latency(model, test_loader, device)
    print(f"⚡ Середня затримка (latency): {avg_latency:.2f} мс / приклад")


# === 4. Точка входу ===
if __name__ == "__main__":
    evaluate_model()                      # Запуск оцінки моделі
