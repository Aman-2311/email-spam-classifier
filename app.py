import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import sklearn
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()  # 1
    text = nltk.word_tokenize(text)  # 2text converted in list so start looping

    y = []  # 3
    for i in text:
        if i.isalnum():  # alpha-numeric(alnum)
            y.append(i)

    text = y[:]  # 4
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
cv = pickle.load(open('vectorizer (2).pkl','rb'))
model = pickle.load(open('spam_model.pkl','rb'))


st.title ("Email/SMS Spam Classifier ")

input_sms = st.text_input("Enter the message")


if st.button('Predict'):

#1. preprocessing
   transformed_sms = transform_text(input_sms)
   vector_input = cv.transform([transformed_sms])
   result = model.predict(vector_input)[0]

   if result ==1:
      st.header("spam")
   else:
      st.header("NOT SPAM")