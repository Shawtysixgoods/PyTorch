
# Руководство по PyTorch: От Основ до Нейронных Сетей

## Введение: Что такое PyTorch?

PyTorch — это мощная библиотека для машинного обучения на Python. Представьте её как универсальный конструктор для создания нейронных сетей. Главная особенность PyTorch — простота и гибкость. Он позволяет легко экспериментировать и быстро создавать прототипы.

## Часть 1: Тензоры — Основа всего

### Что такое тензор?
Тензор — это многомерный массив чисел. Не пугайтесь термина — это просто обобщение понятий, которые вы уже знаете:

- **0D тензор (скаляр)**: одно число — `5.0`
- **1D тензор (вектор)**: список чисел — `[1, 2, 3]`
- **2D тензор (матрица)**: таблица чисел — картинка в оттенках серого
- **3D тензор**: цветная картинка (высота × ширина × цвет)

### Создание тензоров

```python
import torch

# Создание тензоров различными способами
scalar = torch.tensor(7.0)
vector = torch.tensor([1.0, 2.0, 3.0])
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

# Случайные тензоры (полезно для весов нейросети)
random_tensor = torch.rand(2, 3)  # 2×3 матрица случайных чисел от 0 до 1
zeros_tensor = torch.zeros(3, 3)  # 3×3 матрица нулей
ones_tensor = torch.ones(2, 4)    # 2×4 матрица единиц

print(f"Форма матрицы: {matrix.shape}")
print(f"Тип данных: {matrix.dtype}")
```

### Основные операции

```python
# Создадим два тензора для примера
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# Поэлементные операции
print("Сложение:", a + b)
print("Поэлементное умножение:", a * b)

# Матричное умножение (важно для нейросетей!)
print("Матричное умножение:", torch.matmul(a, b))
# Или короче:
print("Матричное умножение:", a @ b)
```

### Изменение формы тензоров

```python
# Создаем тензор и меняем его форму
x = torch.arange(12, dtype=torch.float32)  # [0, 1, 2, ..., 11]
print("Исходный:", x.shape)

# Превращаем в матрицу 3×4
x_reshaped = x.reshape(3, 4)
print("После reshape:", x_reshaped.shape)

# Добавляем/убираем размерности
x_expanded = x_reshaped.unsqueeze(0)  # Добавляем размерность в начало
print("С дополнительной размерностью:", x_expanded.shape)  # [1, 3, 4]
```

## Часть 2: Автоматическое дифференцирование

Это "мозг" PyTorch, который позволяет нейросетям учиться. Когда вы помечаете тензор как `requires_grad=True`, PyTorch начинает отслеживать все операции с ним.

### Простой пример обучения

```python
# Допустим, мы хотим найти число w, такое что w * 2 = 6
# Очевидно, ответ w = 3, но пусть компьютер найдет это сам

w = torch.tensor(1.0, requires_grad=True)  # Начальная догадка
target = torch.tensor(6.0)  # Целевое значение

# Обучение в цикле
learning_rate = 0.1
for step in range(10):
    # Прямой проход: делаем предсказание
    prediction = w * 2
    
    # Вычисляем ошибку (функцию потерь)
    loss = (prediction - target) ** 2
    
    print(f"Шаг {step}: w = {w.item():.3f}, предсказание = {prediction.item():.3f}, потеря = {loss.item():.3f}")
    
    # Обратный проход: вычисляем градиент
    loss.backward()
    
    # Обновляем w в направлении, уменьшающем ошибку
    with torch.no_grad():
        w -= learning_rate * w.grad
    
    # ВАЖНО: обнуляем градиент для следующей итерации
    w.grad.zero_()

print(f"Итоговое значение w: {w.item():.3f}")
```

## Часть 3: Создание нейронных сетей с nn.Module

### Основы nn.Module

`nn.Module` — это базовый класс для всех нейронных сетей в PyTorch. У него два главных метода:

- `__init__()` — здесь мы определяем слои нашей сети
- `forward()` — здесь мы описываем, как данные проходят через сеть

### Простая нейронная сеть

```python
import torch.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        # Определяем слои нашей сети
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()  # Функция активации
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Описываем прохождение данных через сеть
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

# Создаем модель
model = SimpleNetwork(input_size=10, hidden_size=20, output_size=1)
print(model)

# Проверяем на случайных данных
test_input = torch.randn(5, 10)  # 5 примеров, каждый размера 10
output = model(test_input)
print(f"Форма выхода: {output.shape}")
```

### Популярные слои

```python
# Основные строительные блоки
linear_layer = nn.Linear(10, 5)      # Полносвязный слой
activation = nn.ReLU()               # Функция активации
dropout = nn.Dropout(0.2)            # Регуляризация (отключает 20% нейронов)
flatten = nn.Flatten()               # Превращает многомерный тензор в вектор

# Можно объединить в последовательность
network = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10)
)
```

## Часть 4: Работа с данными — Dataset и DataLoader

### Зачем нужен DataLoader?

Представьте, что у вас есть миллион фотографий. Загрузить их все в память сразу невозможно. DataLoader решает эту проблему, подавая данные маленькими порциями (батчами).

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Загружаем датасет MNIST (рукописные цифры)
transform = transforms.Compose([
    transforms.ToTensor(),                    # Превращаем картинку в тензор
    transforms.Normalize((0.1307,), (0.3081,))  # Нормализуем данные
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Создаем DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,      # Размер батча
    shuffle=True,       # Перемешиваем данные
    num_workers=2       # Количество процессов для загрузки
)

# Смотрим на один батч
for images, labels in train_loader:
    print(f"Форма батча изображений: {images.shape}")
    print(f"Форма батча меток: {labels.shape}")
    break  # Смотрим только первый батч
```

## Часть 5: Полный цикл обучения

### Пошаговый алгоритм

1. **Подготовка**: загружаем данные, создаем модель, выбираем функцию потерь и оптимизатор
2. **Цикл обучения**: для каждого батча делаем прямой проход, вычисляем потери, делаем обратный проход, обновляем веса
3. **Оценка**: проверяем качество модели на тестовых данных

### Полный пример

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Подготовка данных
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. Создание модели
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

model = MNISTClassifier()

# 3. Выбор функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()  # Для классификации
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Функция обучения
def train_epoch(model, train_loader, criterion, optimizer):
    model.train()  # Переводим в режим обучения
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Обнуляем градиенты
        optimizer.zero_grad()
        
        # Прямой проход
        output = model(data)
        loss = criterion(output, target)
        
        # Обратный проход и обновление весов
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Батч {batch_idx}, Потеря: {loss.item():.6f}')
    
    return total_loss / len(train_loader)

# 5. Функция тестирования
def test_model(model, test_loader, criterion):
    model.eval()  # Переводим в режим оценки
    test_loss = 0
    correct = 0
    
    with torch.no_grad():  # Отключаем вычисление градиентов
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            
            # Подсчитываем правильные предсказания
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Точность: {accuracy:.2f}%')
    return accuracy

# 6. Основной цикл обучения
num_epochs = 5

for epoch in range(num_epochs):
    print(f'\nЭпоха {epoch+1}/{num_epochs}')
    
    # Обучение
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    print(f'Средняя потеря при обучении: {train_loss:.6f}')
    
    # Тестирование
    accuracy = test_model(model, test_loader, criterion)

print('\nОбучение завершено!')

# 7. Сохранение модели
torch.save(model.state_dict(), 'mnist_model.pth')
print('Модель сохранена!')
```

## Важные советы для начинающих

### 1. Выбор устройства (CPU vs GPU)
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Используется устройство: {device}')

# Отправляем модель и данные на устройство
model = model.to(device)
data = data.to(device)
```

### 2. Сохранение и загрузка моделей
```python
# Сохранение
torch.save(model.state_dict(), 'model.pth')

# Загрузка
model = MNISTClassifier()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### 3. Общие ошибки и как их избежать

**Забыли обнулить градиенты:**
```python
# НЕПРАВИЛЬНО
for data, target in train_loader:
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()  # Градиенты накапливаются!

# ПРАВИЛЬНО
for data, target in train_loader:
    optimizer.zero_grad()  # Обнуляем градиенты
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

**Неправильный режим модели:**
```python
# При обучении
model.train()

# При тестировании или использовании
model.eval()
```

**Неправильная форма данных:**
```python
# Проверяйте формы тензоров
print(f"Форма входных данных: {data.shape}")
print(f"Форма выходных данных: {output.shape}")
print(f"Форма меток: {target.shape}")
```

## Что изучать дальше?

1. **Сверточные нейронные сети (CNN)** — для работы с изображениями
2. **Рекуррентные сети (RNN, LSTM)** — для работы с последовательностями
3. **Трансформеры** — современная архитектура для NLP и не только
4. **Трансферное обучение** — использование предобученных моделей
5. **Продвинутые техники** — регуляризация, оптимизация, аугментация данных

## Полезные ресурсы

- [Официальная документация PyTorch](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Papers With Code](https://paperswithcode.com/) — для изучения современных архитектур

