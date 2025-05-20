import joblib
import re 
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tkinter as tk
from tkinter import messagebox
from textblob import TextBlob


model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

lemmatizer = WordNetLemmatizer()
def cleaning_data (text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '',text)
    blob = TextBlob(text)
    blob = str(blob.correct())
    blob = TextBlob(blob)
    clean_tokens= []
    for word , tag in blob.tags : 
        if word not in stopwords.words('english'):
            if tag.startswith("V"):
                lemma = lemmatizer.lemmatize(word , pos = 'v')
            elif tag.startswith("J"):
                lemma = lemmatizer.lemmatize(word , pos = 'a')
            elif tag.startswith("R"):
                lemma = lemmatizer.lemmatize(word, pos = 'r')
            else:
                lemma = lemmatizer.lemmatize(word)
            clean_tokens.append(lemma)
    return ' '.join(clean_tokens)



def predicting():
    message = entry.get('1.0',tk.END).strip()
    if not message:
        messagebox.showwarning('input error','pls enter a messge.')
        return
    cleansed = cleaning_data(message)
    print(cleansed)
    X_input = vectorizer.transform([cleansed])
    prediction = model.predict(X_input)[0]
    result = 'Spam' if prediction == 'spam' else 'Ham'
    messagebox.showinfo('result:',f"prediction:{result}")




root = tk.Tk()
root.title('Spam Detector')
tk.Label(root,text = "enter your message :").pack()
entry = tk.Text(root, height = 10 , width = 50)
entry.pack()

tk.Button(root,text = "check",command = predicting).pack(pady = 10)
root.mainloop()







