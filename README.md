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

