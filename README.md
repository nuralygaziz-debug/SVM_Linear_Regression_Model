# Прогнозирование успеваемости студентов: Pass / Fail

## Датасет
этот датасет был взят с hugging face.

Почему именно этот датасет я взял для своего проекта?


1.В этом датасете более 1000 строк и 18 признаков (я взял 9) и это подходило для моего проекта.

2.Он не сильно популярен так как в kaggle его установили 478 раз

3.Данные реальные и содержательные

4.Есть дисбаланс классов

5.И самое главное то что он 
не слишком большой и не слишком сложный

## Проект

Главная цель проекта предсказывать сдаст ученик ли тест основываясь 9 признаками которые я выбрал

- `0 = Fail`
- `1 = Pass`

Главная особенность задачи — сильный дисбаланс классов. Поэтому высокая общая `Accuracy` не считается достаточным доказательством хорошей работы модели.

---

## Структура проекта

```text
student-pass-fail-ml-github/
│
├── student_pass_fail_analysis_ru.ipynb
├── README.md
└── requirements.txt
```

---

## Что реализовано

### 1. Baseline

Добавлена простая baseline-модель:

```text
Always predict Pass
```

Она показывает важную проблему несбалансированной классификации: можно получить высокую Accuracy, но не обнаружить ни одного студента класса `Fail`.

---

### 2. Правильные метрики

Для оценки моделей используются:

- Accuracy
- Balanced Accuracy
- Macro-F1
- Fail Precision
- Fail Recall
- Fail F1

Особое внимание уделяется классу `Fail`, потому что именно его сложнее всего обнаружить из-за малого количества примеров.

---

### 3. Разделение данных

Используется один последовательный pipeline:

```text
Train / Validation / Test
≈ 70% / 15% / 15%
```

- Train — обучение моделей.
- Validation — подбор гиперпараметров.
- Test — только финальная оценка.

Разделение выполняется с `stratify`, чтобы сохранить соотношение классов.

---

### 4. Масштабирование

`StandardScaler` обучается только на Train:

```python
scaler.fit(X_train)
```

После этого он применяется к Validation и Test.

Это предотвращает data leakage(это утечка информаций из тестовых данных в обучение модели).

---

### 5. Logistic Regression

Для Logistic Regression исследуются:

```text
C = 0.01, 0.1, 1, 10
С это параметр который определяет силу регуляризации
чем меньше С тем сильнее регуляризация а чем больше С тем
сильнее регуляризация
```

$$
C = \frac{1}{\lambda}
$$

и:

```text
class_weight = None / balanced
class weight это параметр который определяет насколько сильно
модель должна учитывать каждый класс
```
$$
w_i = \frac{n}{k \cdot n_i}
$$

Выбор параметров выполняется по Validation.

---

### 6. Linear SVM

Для SVM используется:

```python
kernel="linear"
```
```text
я использую Soft Margin SVM потому что
датасет имеет шумы и не равномерный график для этого и нужен Soft Margin
а если бы в датасете не было бы шумов я бы взял Hard Margin SVM

```

## SVM Hyperplane

График показывает распределение студентов из датасета и разделяющую гиперплоскость линейной модели SVM.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Dataset
url = "https://huggingface.co/datasets/jason1966/algozee_student/resolve/main/student_performance_interactions.csv"
df = pd.read_csv(url)

# Выбери два числовых признака
X = df[["hours_studied", "previous_score"]]
y = df["passed"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Logistic Regression
model = LogisticRegression(
    C=1,
    class_weight="balanced",
    random_state=42
)
model.fit(X_scaled, y)

# Grid
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

grid = np.c_[xx.ravel(), yy.ravel()]

# Decision function
Z = model.decision_function(grid).reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 7))

# Data points
plt.scatter(
    X_scaled[y == 0, 0],
    X_scaled[y == 0, 1],
    label="Fail",
    alpha=0.7
)

plt.scatter(
    X_scaled[y == 1, 0],
    X_scaled[y == 1, 1],
    label="Pass",
    alpha=0.7
)

# Hyperplane / decision boundary
plt.contour(
    xx,
    yy,
    Z,
    levels=[0],
    linewidths=2
)

plt.xlabel("Hours studied")
plt.ylabel("Previous score")
plt.title("Logistic Regression — Decision Boundary")

plt.legend()
plt.grid(alpha=0.3)

# Save for GitHub
plt.savefig("hyperplane.png", dpi=300, bbox_inches="tight")

plt.show()
```
## Decision Boundary

The graph shows the data points from the dataset and the decision boundary of the Logistic Regression model.

![Decision Boundary](hyperplane.png)

Это соответствует линейной soft-margin SVM.

Также исследуются разные значения:

```text
C = 0.01, 0.1, 1, 10
```

и варианты balancing классов.

---

### 7. Анализ ошибок

Вместо того чтобы ограничиваться только TN, FP, FN и TP, в проекте ошибки описываются напрямую:

- Фактически Fail → Предсказано Fail
- Фактически Fail → Предсказано Pass
- Фактически Pass → Предсказано Fail
- Фактически Pass → Предсказано Pass

Наиболее опасная ошибка:

```text
Фактически Fail → Предсказано Pass
```

Это означает, что модель пропустила студента группы риска.

---

### 8. Проверка совпадения предсказаний моделей

В проект добавлена отдельная проверка:

```python
pred_logistic == pred_svm
```

Если Logistic Regression и Linear SVM дают одинаковые результаты для всех объектов Test, это выводится отдельно и интерпретируется.

---

### 9. Feature selection experiment

Сравниваются два набора признаков:

#### Только академические признаки

- previous scores
- study hours
- attendance
- homework completion

#### Академические + образ жизни

Дополнительно:

- sleep hours
- screen time

Исследовательский вопрос:

> Помогает ли информация об образе жизни улучшить прогнозирование академического риска?

---

### 10. Probability threshold experiment

Для Logistic Regression используется `predict_proba()`.

Probability threshold выбирается по Validation, после чего оценивается на Test.

Это позволяет исследовать компромисс между:

- высоким `Fail Recall`;
- высоким `Fail Precision`.

---

## Главный вывод проекта

Для несбалансированной задачи высокая Accuracy не гарантирует хорошую модель.

Baseline может показывать высокую Accuracy, но при этом не находить ни одного студента класса `Fail`.

Модель с balancing классов может иметь более низкую Accuracy, но значительно лучший `Fail Recall`.

Например, ситуация:

```text
Высокий Fail Recall
+
Низкий Fail Precision
```

означает:

- модель хорошо находит студентов группы риска;
- но одновременно создаёт много ложных тревог.

Поэтому итоговая оценка должна учитывать компромисс между Precision и Recall.

---

## Ограничения проекта

В датасете относительно мало студентов класса `Fail`.

После разделения Train / Validation / Test в Validation и Test остаётся примерно 10–11 объектов класса `Fail`.

Из-за этого выбор гиперпараметров по одному Validation split может быть нестабильным.

В будущей версии проекта рекомендуется использовать:

```text
Stratified 5-Fold Cross-Validation
```

для выбора гиперпараметров, после чего выполнять одну финальную оценку на отдельном Test-наборе.

---

## Как запустить

### Вариант 1. Google Colab

1. Открыть Google Colab.
2. Загрузить файл `student_pass_fail_analysis_ru.ipynb`.
3. Выполнить:

```text
Runtime → Restart session and run all
```

### Вариант 2. Локально

Установить зависимости:

```bash
pip install -r requirements.txt
```

Затем открыть notebook:

```bash
jupyter notebook
```

---

## Датасет

В проекте используется CSV-датасет, загружаемый напрямую из Hugging Face.

Источник данных указан в самом notebook.
