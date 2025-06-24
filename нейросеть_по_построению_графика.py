import torch
import torch.nn as nn  # Основные компоненты нейросетей
import numpy as np     # Для работы с массивами
import matplotlib.pyplot as plt  # Для визуализации

# Определяем простейшую нейронную сеть
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Создаем один линейный слой (фактически линейную регрессию)
        # nn.Linear(1, 1) означает:
        # - 1 входной признак (x)
        # - 1 выходное значение (y)
        # Внутри автоматически создаются:
        # - weight (вес) - коэффициент наклона
        # - bias (смещение) - свободный член
        self.layer = nn.Linear(1, 1)

    def forward(self, x):
        # Прямой проход - просто применяем линейный слой
        return self.layer(x)

def main():
    print("=== Простая нейросеть с графиком ===")

    # 1. ПОДГОТОВКА ДАННЫХ ================================================
    
    # Создаем искусственные данные с линейной зависимостью y = 2x + 1 + шум
    x_data = torch.linspace(-2, 2, 50).reshape(-1, 1)  # 50 точек от -2 до 2
    y_data = 2 * x_data + 1 + 0.1 * torch.randn_like(x_data)  # Добавляем шум
    
    # 2. СОЗДАНИЕ МОДЕЛИ И НАСТРОЙКИ ОБУЧЕНИЯ ============================
    
    model = SimpleNet()  # Инициализируем модель
    
    # Функция потерь - среднеквадратичная ошибка (MSE)
    # Измеряет, насколько предсказания отличаются от истинных значений
    criterion = nn.MSELoss()
    
    # Оптимизатор - стохастический градиентный спуск (SGD)
    # lr=0.01 - скорость обучения (learning rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # 3. ПРОЦЕСС ОБУЧЕНИЯ ================================================
    
    print("Обучение...")
    for epoch in range(200):  # 200 эпох обучения
        # Прямой проход: получаем предсказания
        pred = model(x_data)
        
        # Вычисляем ошибку
        loss = criterion(pred, y_data)
        
        # Обратный проход
        optimizer.zero_grad()  # Обнуляем градиенты
        loss.backward()        # Вычисляем градиенты
        optimizer.step()       # Обновляем параметры
        
        # Выводим прогресс каждые 50 эпох
        if epoch % 50 == 0:
            print(f'Эпоха {epoch}, Потери: {loss:.3f}')
    
    # 4. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ========================================
    
    model.eval()  # Переводим модель в режим оценки
    with torch.no_grad():  # Отключаем вычисление градиентов для экономии памяти
        # Создаем тестовые данные для гладкой линии предсказания
        x_test = torch.linspace(-3, 3, 100).reshape(-1, 1)
        y_pred = model(x_test)  # Получаем предсказания
        
        # Создаем график
        plt.figure(figsize=(8, 6))
        
        # Исходные данные (синие точки)
        plt.scatter(x_data.numpy(), y_data.numpy(), alpha=0.6, label='Данные')
        
        # Предсказания модели (красная линия)
        plt.plot(x_test.numpy(), y_pred.numpy(), 'r-', label='Предсказание нейросети')
        
        # Настройки графика
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Простая нейросеть - линейная регрессия')
        plt.legend()
        plt.grid(True)
        
        # Сохраняем график в файл
        plt.savefig('neural_network_result.png', dpi=300, bbox_inches='tight')
        print("График сохранен в файл: neural_network_result.png")
        plt.close()  # Закрываем фигуру для освобождения памяти

    # 5. АНАЛИЗ РЕЗУЛЬТАТОВ ==============================================
    
    # Выводим обученные параметры
    for name, param in model.named_parameters():
        print(f"{name}: {param.data.numpy()}")
    
    print("Обучение завершено!")

if __name__ == "__main__":
    main()