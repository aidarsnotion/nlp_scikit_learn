# NLP.py - Предобработка текста с использованием Tokenizer и NLTK

Этот проект демонстрирует полный процесс предобработки текста, превращая текстовые данные в числовые последовательности, которые могут использоваться в моделях машинного обучения, в частности, в нейронных сетях. Процесс включает токенизацию, удаление стоп-слов и чисел, а также дополнение последовательностей для обеспечения одинаковой длины.

## Возможности

- **Очистка текста**: Преобразование текста в нижний регистр, удаление стоп-слов и неалфавитных токенов (например, чисел и символов).
- **Токенизация**: Преобразование очищенного текста в числовые последовательности с использованием `Tokenizer` из TensorFlow.
- **Дополнение последовательностей**: Обеспечение одинаковой длины всех последовательностей с помощью функции `pad_sequences`.

## Требования

Для работы скрипта необходимы следующие библиотеки Python:

- `tensorflow` (для `Tokenizer` и `pad_sequences`)
- `nltk` (для токенизации и удаления стоп-слов)

Установите зависимости с помощью команды:

```bash
pip install tensorflow nltk
```

# Пример:

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


# Требуемые зависимости
Перед запуском убедитесь, что установлены все необходимые библиотеки. Список необходимых пакетов в файле requirements.txt в корне проекта.

Все пакеты из requirements.txt можно установить одним махом: 
```
pip install -r requirements.txt
```

