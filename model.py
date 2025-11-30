# model.py
"""
МОДУЛЬ: model.py
=================
Призначення:
------------
- Визначає архітектуру простої CNN-моделі для класифікації аудіо-команд
- Приймає на вхід мел-спектрограму (2D-зображення)
- Вихід: клас (yes / no / up / down)
"""

import torch                     # Імпорт бібліотеки PyTorch для роботи з тензорами
import torch.nn as nn            # Імпорт модулю для створення нейромережевих шарів
import torch.nn.functional as F  # Імпорт функцій активації та інших функцій


# === 1. Архітектура моделі ===
class SpeechCommandCNN(nn.Module):          # Оголошення класу моделі
    def __init__(self, num_classes=4):      # Конструктор класу, параметр — кількість класів
        super(SpeechCommandCNN, self).__init__()  # Виклик конструктора батьківського класу

        # Вхідна форма спектрограми ≈ [1, 64, ~100] (1 канал, 64 мел-фільтри)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Перший згортковий шар
        self.bn1 = nn.BatchNorm2d(16)         # Батч-нормалізація після першої згортки
        self.pool1 = nn.MaxPool2d(2, 2)       # Пулінг для зменшення розміру фіч-карти

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Друга згортка
        self.bn2 = nn.BatchNorm2d(32)         # Нормалізація після другої згортки
        self.pool2 = nn.MaxPool2d(2, 2)       # Ще один MaxPool для зменшення розміру

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Третя згортка
        self.bn3 = nn.BatchNorm2d(64)         # Нормалізація після третьої згортки
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))  # Адаптивний пулінг до розміру 1×1

        # Вихід CNN перетворюємо у вектор
        self.fc = nn.Linear(64, num_classes)  # Фінальний повнозв’язний шар для класифікації

    def forward(self, x):                     # Метод forward — шлях проходження даних через модель
        # Очікуємо вхід [batch, 1, mel_bins, time]
        x = F.relu(self.bn1(self.conv1(x)))   # Перша згортка + нормалізація + ReLU
        x = self.pool1(x)                     # Пулінг

        x = F.relu(self.bn2(self.conv2(x)))   # Друга згортка + нормалізація + ReLU
        x = self.pool2(x)                     # Пулінг

        x = F.relu(self.bn3(self.conv3(x)))   # Третя згортка + нормалізація + ReLU
        x = self.pool3(x)                     # Адаптивний пулінг

        x = torch.flatten(x, 1)               # Перетворення у вектор (batch, 64)
        x = self.fc(x)                        # Класифікація через лінійний шар
        return x                              # Повертаємо логіти (оцінки для кожного класу)


# === 2. Тест (запуск напряму) ===
if __name__ == "__main__":                   # Виконується тільки при прямому запуску файлу
    model = SpeechCommandCNN(num_classes=4)   # Створення моделі
    sample = torch.randn(8, 1, 64, 100)       # Вхідний випадковий тензор (8 прикладів)
    out = model(sample)                       # Пропускаємо дані через модель
    print("Форма виходу:", out.shape)         # Виводимо форму результату
