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
     - `ngram_range=(1, 2)`: Униграммы и биграммы.
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
```python
review = 'The long lines and poor customer service really turned me off.'
probability = model.predict_proba(vectorizer.transform([review]))[0][1]
print(f"Вероятность положительного отзыва: {probability:.2f}")
```


#

# Text Preprocessing with Tokenizer and NLTK

This project demonstrates a complete text preprocessing pipeline for converting textual data into numerical sequences, which can be used as input for machine learning models, particularly neural networks. The process includes tokenization, removal of stop-words and numbers, and padding sequences to ensure uniform length.

## Features

- **Text Cleaning**: Converts text to lowercase, removes stop-words, and filters out non-alphabetic tokens (e.g., numbers, symbols).
- **Tokenization**: Converts cleaned text into sequences of numerical indices using TensorFlow's `Tokenizer`.
- **Padding**: Ensures all sequences have the same length using `pad_sequences`.

## Requirements

The script uses the following Python libraries:

- `tensorflow` (for `Tokenizer` and `pad_sequences`)
- `nltk` (for tokenization and stop-word removal)

Install the dependencies using:

```bash
pip install tensorflow nltk

