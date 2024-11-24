# NLP.py - Предобработка текста с использованием Tokenizer и NLTK

## Возможности

- **Очистка текста**: Преобразование текста в нижний регистр, удаление стоп-слов и неалфавитных токенов (например, чисел и символов).
- **Токенизация**: Преобразование очищенного текста в числовые последовательности с использованием `Tokenizer` из TensorFlow.
- **Дополнение последовательностей**: Обеспечение одинаковой длины всех последовательностей с помощью функции `pad_sequences`.

## Требования

Для работы скрипта необходимы следующие библиотеки Python:

- `tensorflow` (для `Tokenizer` и `pad_sequences`)
- `nltk` (для токенизации и удаления стоп-слов)

# Пример раблоты:

Входдной текст:
```
lines = [
    "The quick brown fox",
    "Jumps over $$$ the lazy brown dog",
    "Who jumps high into the blue sky after counting 123",
    "And quickly returns to earth"
]
```
Обработанный текст:
```
['quick brown fox', 'jumps lazy brown dog', 'jumps high blue sky counting', 'quickly returns earth']

```

Индексы слов:

```
{'quick': 1, 'brown': 2, 'fox': 3, 'jumps': 4, 'lazy': 5, 'dog': 6, 'high': 7, 'blue': 8, 'sky': 9, 'quickly': 10, 'returns': 11, 'earth': 12}
```

Числовые последовательности:
```
[[3, 1, 4], [2, 5, 1, 6], [2, 7, 8, 9, 10], [11, 12, 13]]
```

Выравненные последовательности:
```
[
 [ 3  1  4  0  0]
 [ 2  5  1  6  0]
 [ 2  7  8  9 10]
 [11 12 13  0  0]
]
```

# s_analysis.py - Анализ тональности отзывов с использованием Logistic Regression

## Используемые технологии и библиотеки
- **Python**: Основной язык программирования.
- **Pandas**: Для загрузки и анализа данных.
- **Scikit-learn**: Для обработки текстовых данных, построения модели и оценки результатов.
- **Matplotlib**: Для визуализации матрицы ошибок (через ConfusionMatrixDisplay).

##
1. **Загрузка данных**:
   - Данные загружаются из CSV-файла `reviews.csv`.
   - Ожидается два столбца:
     - `Text`: Текст отзыва.
     - `Sentiment`: Метка тональности (`Positive` или `Negative`).

2. **Предобработка данных**:
   - Удаление дубликатов записей для предотвращения переобучения.
   - Анализ распределения данных по тональности с помощью методов `info()` и `groupby()`.
  
3. **Векторизация текста**:
   - Используется `CountVectorizer` для преобразования текста в числовые представления.
   - Применяются следующие параметры:
     - `ngram_range=(1, 2)`: Униграммы и биграммы. Униграммы (n=1) — отдельные слова. Например, из фразы "The food was great" будут выделены слова: ["The", "food", "was", "great"]. Биграммы (n=2) — последовательности из двух слов. Например, из фразы "The food was great" будут выделены биграммы: ["The food", "food was", "was great"].
     - `stop_words='english'`: Исключение часто встречающихся слов (стоп-слов).
     - `min_df=20`: Исключение редких слов (встречающихся менее чем в 20 документах).

4. **Обучение модели**:
   - Тексты разделяются на обучающую и тестовую выборки в соотношении 50/50.
   - Используется модель логистической регрессии (`LogisticRegression`).

5. **Оценка модели**:
   - Построение и визуализация матрицы ошибок для тестовой выборки.
   - Проверка модели на новых отзывах с предсказанием вероятности тональности.

## Пример работы
Пример ввода нового текста и анализа его тональности:

Оценка вероятности тональности для нового отзыва
```python
# Положительные примеры
positive_texts = [
    "The food was delicious and the staff was very friendly.",
    "I loved the ambiance, and the service was excellent!"
]

for text in positive_texts:
    prob = model.predict_proba(vectorizer.transform([text]))[0][1]
    print(f"Text: '{text}' -> Positive probability: {prob}")

# Отрицательные примеры
negative_texts = [
    "The customer service was rude and the food was undercooked.",
    "The room was dirty, and I had to wait for hours to get assistance."
]

for text in negative_texts:
    prob = model.predict_proba(vectorizer.transform([text]))[0][1]
    print(f"Text: '{text}' -> Positive probability: {prob}")
```

Результат анализа:
```
Text: 'The food was delicious and the staff was very friendly.' -> Positive probability: 0.5791591375349925
Text: 'I loved the ambiance, and the service was excellent!' -> Positive probability: 0.8430843825114044
Text: 'The customer service was rude and the food was undercooked.' -> Positive probability: 0.3708245366954655
Text: 'The room was dirty, and I had to wait for hours to get assistance.' -> Positive probability: 0.4325179343151567
```

# spam_analysis.py Spam Detection

### 1. Загрузка и очистка данных
- Загрузка данных из CSV-файла.
- Очистка текстов:
  - Приведение текста к нижнему регистру.
  - Удаление пунктуации и чисел.
  - Замена невалидных значений (например, `NaN`) на пустые строки.
- Удаление дубликатов из набора данных.

### 2. Балансировка данных
- Преобразование текста в числовую форму с использованием **TF-IDF** векторизации.
  - Поддержка униграмм и биграмм.
  - Исключение часто и редко встречающихся слов.
- Балансировка данных с использованием **SMOTE** (Synthetic Minority Oversampling Technique).

### 3. Разделение данных
- Данные разделяются на обучающую и тестовую выборки в соотношении 70% и 30%.

### 4. Обучение модели
- Используется **Random Forest Classifier**:
  - 100 деревьев решений.
  - Случайное состояние фиксировано для воспроизводимости.

### 5. Оценка качества модели
- **ROC-кривая**:
  - Построение кривой.
  - Расчёт значения AUC (Area Under the Curve) для оценки модели.
- **Матрица ошибок**:
  - Анализ предсказаний на тестовой выборке.

### 6. Предсказание
- Предоставляется функция `predict_spam`, которая принимает текст сообщения и возвращает вероятность того, что сообщение является спамом.

---

## Пример работы:

```
message_1 = 'Congratulations! You have won a $1000 gift card. Click here to claim.'
message_2 = '''
Hello John,
I hope this message finds you well. I just wanted to follow up on our meeting last week regarding the project timeline. 
Looking forward to your reply.
'''
```

```
Вероятность спама для сообщения 1: 0.9700
Вероятность спама для сообщения 2: 0.5000
```

# Требуемые зависимости
Перед запуском убедитесь, что установлены все необходимые библиотеки. Список необходимых пакетов в файле requirements.txt в корне проекта.

Все пакеты из requirements.txt можно установить одним махом: 
```
pip install -r requirements.txt
```

