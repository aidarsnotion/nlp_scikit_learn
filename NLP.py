# Импортируем необходимые библиотеки
from tensorflow.keras.preprocessing.text import Tokenizer  # Для токенизации текста
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Для выравнивания последовательностей
from nltk.tokenize import word_tokenize  # Для токенизации слов
from nltk.corpus import stopwords  # Для удаления стоп-слов
import nltk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

# Загрузим ресурсы для библиотеки NLTK
nltk.download('punkt')  # Для токенизации текста / вызвал ошибку, скачивание punkt_tab решило ошибку
nltk.download('stopwords')  # Для работы со стоп-словами
nltk.download('punkt_tab') # Необходимые ресурсы для токенизации текста

# Исходные данные: строки текста
lines = [
    "The quick brown fox",
    "Jumps over $$$ the lazy brown dog",
    "Who jumps high into the blue sky after counting 123",
    "And quickly returns to earth"
]

# Функция для удаления стоп-слов и символов
def preprocess_text(text):
    text = word_tokenize(text.lower())  # Перевод текста в нижний регистр и токенизация
    stop_words = set(stopwords.words('english'))  # Получение набора стоп-слов
    text = [word for word in text if word.isalpha() and word not in stop_words]  # Удаляем стоп-слова и цифры
    return ' '.join(text)

# Применяем предобработку к каждой строке
processed_lines = list(map(preprocess_text, lines))

# Инициализируем токенизатор и обучаем его на обработанном тексте
tokenizer = Tokenizer()
tokenizer.fit_on_texts(processed_lines)

# Преобразуем текст в последовательности индексов
sequences = tokenizer.texts_to_sequences(processed_lines)

# Выравниваем последовательности до фиксированной длины (например, до 5)
max_length = 5
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Словарь токенов
word_index = tokenizer.word_index

# Результаты
print("Обработанный текст:", processed_lines)
print("Словарь токенов:", word_index)
print("Последовательности:", sequences)
print("Выравненные последовательности:")
print(padded_sequences)


# Параметры
vocab_size = 10000  # Размер словаря
embedding_dim = 32  # Размерность эмбеддингов
sequence_length = 100  # Длина последовательностей

# Создаем модель
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length),
    Flatten(),  # Преобразует выход Embedding слоя в одномерный вектор
    Dense(1, activation='sigmoid')  # Пример: бинарная классификация
])

# Компилируем модель
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Выводим архитектуру
model.summary()