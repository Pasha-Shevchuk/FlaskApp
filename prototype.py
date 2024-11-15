import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
import joblib

LogReg_model = joblib.load('LogisticRegression.pkl')
DesTree_model = joblib.load("DecisionTree.pkl")
GradBoostingClassifier_model = joblib.load('GradientBoostingClassifier.pkl')
RandForClassifier_model = joblib.load('RandomForestClassifier.pkl')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



def better_text_processing(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs and HTML tags (in one step)
    text = re.sub(r'http\S+|www\S+|<.*?>', '', text)

    # Remove square brackets and content inside
    text = re.sub(r'\[.*?\]', '', text)

    # Remove non-word characters
    text = re.sub(r'\W', ' ', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Remove punctuation and alphanumeric words
    tokens = [word for word in tokens if word.isalpha()]

    # Join tokens back into a string
    text = ' '.join(tokens)

    return text






from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = joblib.load('vectorization.pkl')

def ResultsEvaluation(news:str):
    news_testing = {"text":[news]}
    sam_def_test = pd.DataFrame(news_testing) # sam stands for sample !!! 
    sam_def_test["text"] = sam_def_test["text"].apply(better_text_processing) 
    sam_x_test = sam_def_test["text"]
    sam_xv_test = vectorization.transform(sam_x_test)

    pred_RFC = RandForClassifier_model.predict(sam_xv_test)
    pred_LR = LogReg_model.predict(sam_xv_test)
    pred_GBC = GradBoostingClassifier_model.predict(sam_xv_test)
    pred_DT = DesTree_model.predict(sam_xv_test)

    scoreRes = [pred_LR[0],pred_DT[0],pred_GBC[0],pred_RFC[0]]

    return f"\n\nLR Prediction: {pred_LR[0]} \nDT Prediction: {pred_DT[0]} \nGBC Prediction: {pred_GBC[0]} \nRFC Prediction: {pred_RFC[0]}",scoreRes.count(1)

def PrintResults(res:tuple[str:int]):
    print(res[0])
    print(f"score is: {res[1]}/4")





import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



from googletrans import Translator

def translate_to_english(text, target_language='en'):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text


news = str(input())
news = translate_to_english(news)
res = ResultsEvaluation(news)
PrintResults(res)
