from zipfile import ZipFile
import pandas as pd
import numpy as np
from collections import Counter
import warnings
import re
from indicnlp.tokenize import indic_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import ttk

warnings.filterwarnings("ignore")

df_sarcastic = pd.read_csv('sarcasmh/Sarcasm_Hindi_Tweets-SARCASTIC.csv')
df_non_sarcastic = pd.read_csv('sarcasmh/Sarcasm_Hindi_Tweets-NON-SARCASTIC.csv')

# Preprocess the data
df_sarcastic['label'] = 'sarcastic'
df_non_sarcastic['label'] = 'non_sarcastic'
df = pd.concat([df_sarcastic, df_non_sarcastic], axis=0)
df = df.drop(['username', 'acctdesc', 'location', 'following', 'followers', 'totaltweets', 'usercreatedts', 'tweetcreatedts', 'retweetcount', 'hashtags'], axis=1)
df = df.reset_index()
df = df.drop('index', axis=1)

# Function to count the word count
def count_length():
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split(" ")))
count_length()

# Remove emojis from the text
emoji_pattern = re.compile("["
        u"U0001F600-U0001F64F"  # emoticons
        u"U0001F300-U0001F5FF"  # symbols & pictographs
        u"U0001F680-U0001F6FF"  # transport & map symbols
        u"U0001F1E0-U0001F1FF"  # flags (iOS)
        u"U00002500-U00002BEF"  # chinese char
        u"U00002702-U000027B0"
        u"U00002702-U000027B0"
        u"U000024C2-U0001F251"
        u"U0001f926-U0001f937"
        u"U00010000-U0010ffff"
        u"u2640-u2642"
        u"u2600-u2B55"
        u"u200d"
        u"u23cf"
        u"u23e9"
        u"u231a"
        u"ufe0f"  # dingbats
        u"u3030"
                           "]+", flags=re.UNICODE)
for i in range(len(df)):
    df['text'][i] = emoji_pattern.sub(r'', df['text'][i])
count_length()

# Text preprocessing functions
def processText(text):
    text = text.lower()
    text = re.sub('((www.[^s]+)|(https?://[^s]+))', '', text)
    text = re.sub('@[^s]+', '', text)
    text = re.sub('[s]+', ' ', text)
    text = re.sub(r'#([^s]+)', r'1', text)
    return text

df['text'] = df['text'].apply(processText)

# Tokenization and removing Hindi stopwords
def tokenization(indic_string):
    tokens = []
    for t in indic_tokenize.trivial_tokenize(indic_string):
        tokens.append(t)
    return tokens

df['text'] = df['text'].apply(lambda x: tokenization(x))

for i in range(len(df)):
    df['text'][i] = [s.replace("\n", "") for s in df['text'][i]]

stopwords_hi = ['तुम','मेरी','मुझे','क्योंकि','हम','प्रति','अबकी','आगे','माननीय','शहर','बताएं','कौनसी','क्लिक','किसकी','बड़े','मैं','and','रही','आज','लें','आपके',
                'मिलकर','सब','मेरे','जी','श्री','वैसा','आपका','अंदर', 'अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं',
                'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस',
                'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर','करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी',
                'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर',
                'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों',
                'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पर', 'पहले',
                'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या',
                'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वर्ग', 'वह', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'वग़ैरह', 'संग',
                'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते',
                'होना', 'होने', 'अपनि', 'जेसे', 'होति', 'सभि', 'तिंहों', 'इंहों', 'दवारा', 'इसि', 'किंहें', 'थि', 'उंहों', 'ओर', 'जिंहें', 'वहिं', 'अभि', 'बनि', 'हि',
                'उंहिं', 'उंहें', 'हें', 'वगेरह', 'एसे', 'रवासा', 'कोन', 'निचे', 'काफि', 'उसि', 'पुरा', 'भितर', 'हे', 'बहि', 'वहां', 'कोइ', 'यहां', 'जिंहों', 'तिंहें',
                'किसि', 'कइ', 'यहि', 'इंहिं', 'जिधर', 'इंहें', 'अदि', 'इतयादि', 'हुइ', 'कोनसा', 'इसकि', 'दुसरे', 'जहां', 'अप', 'किंहों', 'उनकि', 'भि', 'वरग',
                'हुअ', 'जेसा', 'नहिं']

to_be_removed = stopwords_hi

df['text'] = df['text'].apply(lambda text: [ele for ele in text if ele not in (to_be_removed)])

# Preprocess the text for model training
df['processed_text'] = df['text'].apply(lambda x: ' '.join(x))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['label'], random_state=42)

# Create TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Function to predict sarcasm for input sentence
def predict_sarcasm(input_sentence):
    input_sentence = tokenization(input_sentence)
    input_sentence = [ele for ele in input_sentence if ele not in (to_be_removed)]
    input_sentence = ' '.join(input_sentence)
    input_sentence_tfidf = tfidf_vectorizer.transform([input_sentence])
    predicted_label = model.predict(input_sentence_tfidf)
    return predicted_label[0]

# Function to process input from the tkinter interface
def process_input():
    input_text = input_entry.get()
    sarcasm_label = predict_sarcasm(input_text)
    if sarcasm_label == 'sarcastic':
        result_label.config(text="The Hindi sentence is sarcastic.")
    else:
        result_label.config(text="The Hindi sentence is not sarcastic.")

# Create a tkinter window
root = tk.Tk()
root.title("Hindi Sarcasm Detector")

# Create a tab for input and result
notebook = ttk.Notebook(root)
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text="Sarcasm Detector")

# Input label and entry
input_label = ttk.Label(tab1, text="Enter a Hindi sentence:")
input_label.grid(row=0, column=0, padx=10, pady=10)

# Increase the width of the input entry and set background color
input_entry = ttk.Entry(tab1, width=50)
input_entry.grid(row=0, column=1, padx=10, pady=10)
input_entry.configure(background='light gray')  # Change the background color

# Button to process input
process_button = ttk.Button(tab1, text="Detect Sarcasm", command=process_input)
process_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

# Label to display the result
result_label = ttk.Label(tab1, text="")
result_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

# Set background color for the result_label
result_label.configure(background='light yellow')
notebook.pack()
root.mainloop()

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report")
print(classification_report(y_test, y_pred))