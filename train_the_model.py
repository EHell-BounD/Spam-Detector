import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from wordcloud import WordCloud
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import re 
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import LinearSVC
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
import joblib

# the model
model = MultinomialNB()

#making the data
df = pd.read_csv("spam.csv", encoding = 'latin-1')
df = df[['v1','v2']]
df.columns = ["label","message"]

#cleaning the data
lemmatizer = WordNetLemmatizer()
def cleaning_data (text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '',text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    clean_tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(clean_tokens)
df['cleaned'] = df['message'].apply(cleaning_data)

#vectorizing the data
vectorizer = TfidfVectorizer(min_df = 5 , max_df = 0.9)
X = vectorizer.fit_transform(df['cleaned'])

#modeling the data
X_train , X_test , Y_train , Y_test = train_test_split(X ,df['label'],test_size = 0.2 ,random_state = 42 )
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

trained_model = MultinomialNB()
trained_model.fit(X_train , Y_train)
joblib.dump(trained_model ,'spam_model.pkl')
joblib.dump(vectorizer,'vectorizer.pkl')


#print
#print('Accuracy: ',accuracy_score(Y_test , Y_pred))
#print('\nClassification report:\n',classification_report(Y_test , Y_pred))
#print('\nconfusion matrix:\n',confusion_matrix(Y_test,Y_pred))

#print(df['cleaned'].head())
#print(X.shape)
#print(vectorizer.get_feature_names_out()[:20])


#processing the data
#df.info()
#print('\n')
#print(df['label'].head(),'\n')
#print(df['message'].value_counts())
#print(df.isnull())


#visualization of the data
#sns.countplot(data=df,x='label')
#plt.title('class distribution')
#plt.show()

#generation the word cloud of the data
#all_text = "".join(df.loc[df['label'] == 'ham','message']) 
#all_text = "".join(df.loc[df['label'] == 'spam','message']) 
#all_text = "".join(df['message'])
#wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
#plt.figure(figsize=(10,5))
#plt.imshow(wordcloud,interpolation='bilinear')
#plt.axis('off')
#plt.title("Word cloud of the data")
#plt.show()

