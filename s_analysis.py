import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay as cmd

# Загрузка CSV-файла с отзывами
df = pd.read_csv('./reviews.csv', encoding='ISO-8859-1')
# Загружается датасет, где ожидаются два столбца: 'Text' (текст отзыва) и 'Sentiment' (метка тональности).

print(df.head())  # Вывод первых 5 строк файла для проверки содержимого

df.info()
# Отображает информацию о структуре данных, включая типы данных, количество записей и отсутствие данных.

df.groupby('Sentiment').describe()
# Анализ распределения отзывов по тональности: рассчитываются статистики для каждой категории (Positive, Negative).

df = df.drop_duplicates()
# Удаление дубликатов строк для предотвращения переобучения и увеличения точности модели.

df.groupby('Sentiment').describe()
# Повторный анализ распределения после удаления дубликатов.

# Создание векторизатора для преобразования текстов в числовую форму
vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english', min_df=20)
# ngram_range=(1, 2): Используются униграммы и биграммы.
# stop_words='english': Исключение часто встречающихся слов (например, "the", "is").
# min_df=20: Исключаются термины, встречающиеся менее чем в 20 документах.

x = vectorizer.fit_transform(df['Text'])
# Преобразование текстов из столбца 'Text' в матрицу признаков.

y = df['Sentiment']
# Метки тональности отзывов.

# Демонстрация работы векторизатора на новом тексте
text = vectorizer.transform(['The long l3ines   and; pOOr customer# service really turned me off...123.'])
text = vectorizer.inverse_transform(text)
print(text)
# Преобразование текста в числовую форму и обратное преобразование для проверки.

# Разделение данных на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
# test_size=0.5: Половина данных используется для тестирования.
# random_state=0: Фиксирует случайное разбиение для воспроизводимости.

# Создание и обучение модели логистической регрессии
model = LogisticRegression(max_iter=1000, random_state=0)
# max_iter=1000: Увеличение максимального количества итераций для сходимости модели.
model.fit(x_train, y_train)
# Обучение модели на обучающей выборке.

# Построение матрицы ошибок на тестовой выборке
cmd.from_estimator(model, x_test, y_test,
                   display_labels=['Negative', 'Positive'],
                   cmap='Blues', xticks_rotation='vertical')
# Визуализация матрицы ошибок: показывает, насколько модель точна в предсказаниях.

# Оценка вероятности тональности для нового отзыва
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
