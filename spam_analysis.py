import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, ConfusionMatrixDisplay as cmd
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# === 1. Загрузка и очистка данных ===
def clean_text(text):
    """Функция для очистки текста."""
    if isinstance(text, str):
        text = text.lower()  # Преобразуем текст в нижний регистр
        text = re.sub(r'[^\w\s]', '', text)  # Удаляем пунктуацию
        text = re.sub(r'\d+', '', text)  # Удаляем числа
        return text
    return ''  # Если текст не строка (например, NaN), вернуть пустую строку

# Загружаем CSV-файл
df = pd.read_csv('./spam_or_not_spam.csv', encoding='ISO-8859-1')

# Проверка данных
print(df.head())
df.info()

# Очистка данных
df['cleaned_text'] = df['email'].apply(clean_text)

# Удаление дубликатов
df = df.drop_duplicates()
print(f"После удаления дубликатов: {df.shape}")

# === 2. Балансировка данных ===
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', min_df=10, max_df=0.9)
x = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

# Балансируем данные с помощью SMOTE
x_resampled, y_resampled = SMOTE().fit_resample(x, y)
print(f"Размер сбалансированных данных: {x_resampled.shape}")

# === 3. Разделение на обучающую и тестовую выборки ===
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.3, random_state=0)

# === 4. Обучение модели ===
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(x_train, y_train)

# === 5. Оценка качества модели ===
# ROC-кривая
y_pred_prob = model.predict_proba(x_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC: {auc}")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend()
plt.show()

# Матрица ошибок
cmd.from_estimator(model, x_test, y_test,
                   display_labels=['Not Spam', 'Spam'],
                   cmap='Blues', xticks_rotation='vertical')

# === 6. Предсказание для новых сообщений ===
def predict_spam(message):
    """Функция для предсказания вероятности спама."""
    message_cleaned = clean_text(message)
    message_vectorized = vectorizer.transform([message_cleaned])
    spam_prob = model.predict_proba(message_vectorized)[0][1]
    return spam_prob

# Пример предсказания
message_1 = 'Congratulations! You have won a $1000 gift card. Click here to claim.'

message_2 = '''
Hello John,  
I hope this message finds you well. I just wanted to follow up on our meeting last week regarding the project timeline. Please let me know if you need any additional information or assistance. Looking forward to your reply.  

Best regards,  
Emily
'''
print(f"Вероятность спама для сообщения 1: {predict_spam(message_1):.4f}")
print(f"Вероятность спама для сообщения 2: {predict_spam(message_2):.4f}")
