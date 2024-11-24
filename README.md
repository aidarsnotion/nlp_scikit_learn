# Предобработка текста с использованием Tokenizer и NLTK

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

