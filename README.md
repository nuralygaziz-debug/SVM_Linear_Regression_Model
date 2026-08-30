# Прогнозирование успеваемости студентов: Pass / Fail

## Датасет
Этот датасет был взят с этого сайта https://huggingface.co/datasets/jason1966/algozee_student/blob/main/student_performance_interactions.csv?utm_source=chatgpt.com.
Датасет содержит более 1000 строк и 18 признаков, из которых 9 были отобраны для построения модели.
Данные отобраны по наиболее значимым признакам, которые могут влиять на итоговый результат тестирования студентов.
Датасет имеет относительно небольшую популярность, что снижает вероятность его широкого использования в аналогичных проектах.

## Исследовательский вопрос:

> Помогает ли информация об образе жизни улучшить прогнозирование академической успеваемости?


## Проект

Главная цель проекта предсказывать сдаст ученик ли тест основываясь 9 признаками которые я выбрал

- `0 = Fail`
- `1 = Pass`

Главная особенность задачи — сильный дисбаланс классов. Поэтому высокая общая `Accuracy` не считается достаточным доказательством хорошей работы модели.

---
### Feature selection experiment

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
- 
#### Не отобранные признаки
Мы не брали другие признаки к примеру как anxiety так как их трудно
объективно объяснить в жизни
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

### 2. Метрики

Для оценки моделей используются:

- Accuracy  
- Balanced Accuracy  
- Macro-F1   
- Fail Precision 
- Fail Recall
- Fail F1

  ### Метрики

Accuracy:

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

Balanced Accuracy:

$$
Balanced\ Accuracy = \frac{Recall_{class1} + Recall_{class2}}{2}
$$

Recall: 

$$
Recall = \frac{TP}{TP + FN}
$$

Macro-F1:

$$
Macro\text{-}F1 = \frac{F1_1 + F1_2 + ... + F1_n}{n}
$$

Fail Precision:

$$
Precision_{fail} = \frac{TP_{fail}}{TP_{fail} + FP_{fail}}
$$

Fail Recall:

$$
Recall_{fail} = \frac{TP_{fail}}{TP_{fail} + FN_{fail}}
$$

Fail F1:

$$
F1_{fail} = 2 \cdot \frac{Precision_{fail} \cdot Recall_{fail}}
{Precision_{fail} + Recall_{fail}}
$$

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

$$
X_{scaled} = \frac{X - \mu}{\sigma}
$$

где:
- $X$ — исходное значение
- $\mu$ — среднее значение признака
- $\sigma$ — стандартное отклонение
- $X_{scaled}$ — масштабированное значение

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

## SVM Visualization

The following plot shows the student data points, the linear SVM decision hyperplane, margin boundaries, and support vectors.

![Linear SVM Hyperplane](svm_hyperplane.png)

### 6. Linear SVM

$$
\min_{w,b} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}\max(0,\,1-y_i(w^Tx_i+b))
$$

Для SVM используется:

```python
kernel="linear"
```
```
Я взял Soft Margin SVM потому что
датасет имеет шумы и не равномерный график

```

## SVM Hyperplane

$$
w^T x + b = 0
$$

График показывает распределение студентов из датасета и разделяющую гиперплоскость линейной модели SVM.



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
