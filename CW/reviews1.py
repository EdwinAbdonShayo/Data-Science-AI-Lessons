
import pandas as pd
import numpy as np
import regex as re
from tensorflow.keras.preprocessing.text import Tokenizer
import os
import nltk
from zipfile import ZipFile
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
import string
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
import matplotlib.image as mpimg
from nltk.corpus import wordnet
import random
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout, Dense

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

file_path = "/content/drive/MyDrive/Colab Notebooks/Advanced stuffs coursework/Datasets/Text Based/IMDB Dataset.csv"
df = pd.read_csv(file_path)
df

df.duplicated()
df.duplicated().sum()

df.drop_duplicates(inplace=True, ignore_index=True)

df['review'] = df['review'].str.replace(r'<.*?>', '', regex=True)

df['review'] = df['review'].str.replace(r'[^\w\s]', '', regex=True)

df['review'] = df['review'].str.lower()

nltk.download('wordnet', download_dir='/root/nltk_data')
os.makedirs('/root/nltk_data/corpora', exist_ok=True)
nltk.data.path.append('/root/nltk_data')

with ZipFile('/root/nltk_data/corpora/wordnet.zip', 'r') as zip_ref:
    zip_ref.extractall('/root/nltk_data/corpora')

lemmatizer = WordNetLemmatizer()

df['review'] = df['review'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word, pos='v') for word in x.split()]))

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

df["review"] = df["review"].apply(remove_stopwords)

nltk.download('omw-1.4')

def synonym_replacement(text, n=2):
    words = text.split()
    new_words = words.copy()

    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    if len(random_word_list) == 0:
        return text

    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return " ".join(new_words)

df = pd.concat([df, df.assign(review=df['review'].apply(lambda x: synonym_replacement(x, n=2)))], ignore_index=True)

df

df['sentiment'].value_counts()

df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

df['sentiment'] = df['sentiment'].astype(int)

df.info()

X = df['review']
y = df['sentiment']

max_sequence_len = 0
for sentence in X:
    max_sequence_len = max(len(sentence), max_sequence_len)
print(max_sequence_len)

length_text = [len(text) for text in X]
plt.figure(figsize=(10, 6))
sns.histplot(data=length_text, bins=50, kde=True)
plt.xlabel('Length of Text/Sentence', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Text Length Distribution', fontsize=14)
plt.xlim(0, max(length_text) + 1)
plt.tight_layout()
plt.show()

length_df = pd.DataFrame(length_text)

length_df.describe()

length_df.value_counts()

max_sequence_len = 250

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)

len(X_sequences)

X_padded = pad_sequences(X_sequences, maxlen=max_sequence_len, dtype='float32', padding='post')

X_padded.shape

x_train, x_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    Input(shape=(max_sequence_len,)),
    Embedding(input_dim=10001, output_dim=64),
    Dropout(0.3),
    Bidirectional(LSTM(16)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=5, verbose=1, restore_best_weights='True')
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience=3, factor=0.5, verbose=1, min_lr=0.00001)

model.summary()

plot_model(model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)
img = mpimg.imread("model_plot.png")
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis("off")
plt.show()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, batch_size=64,
                    callbacks=[early_stopping, reduce_lr])

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

y_pred = (model.predict(x_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))

