from flask import Flask, render_template, request
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from googletrans import Translator
from langdetect import detect_langs  # Import language detection library
import joblib

app = Flask(__name__)

LogReg_model = joblib.load('LogisticRegression.pkl')
DesTree_model = joblib.load("DecisionTree.pkl")
GradBoostingClassifier_model = joblib.load('GradientBoostingClassifier.pkl')
RandForClassifier_model = joblib.load('RandomForestClassifier.pkl')
vectorization = joblib.load('vectorization.pkl')

def text_processing(text):
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


def translate_to_english(text, target_language='en'):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text

def ResultsEvaluation(news):
    news_testing = {"text": [news]}
    sam_def_test = pd.DataFrame(news_testing)
    sam_def_test["text"] = sam_def_test["text"].apply(text_processing)
    sam_x_test = sam_def_test["text"]
    sam_xv_test = vectorization.transform(sam_x_test)

    pred_RFC = RandForClassifier_model.predict(sam_xv_test)
    pred_LR = LogReg_model.predict(sam_xv_test)
    pred_GBC = GradBoostingClassifier_model.predict(sam_xv_test)
    pred_DT = DesTree_model.predict(sam_xv_test)

    scoreRes = [pred_LR[0], pred_DT[0], pred_GBC[0], pred_RFC[0]]

    return f"LR Prediction: {pred_LR[0]} \nDT Prediction: {pred_DT[0]} \nGBC Prediction: {pred_GBC[0]} \nRFC Prediction: {pred_RFC[0]}", scoreRes.count(1),scoreRes

def verdict(count):
    verdict_ = ""
    if(count == 0):
        verdict_ = "fake information"
    elif(count == 1):
        verdict_ = "possible fake"
    elif(count == 2 or count == 3):
        verdict_= "information needs to be clarified"
    elif(count == 4):
        verdict_ = "true information"
    else:
        raise(ValueError)
    
    return verdict_

def PrintResults(res):
    # return f"{res[0]}\nscore is: {res[1]}/4,\n persentage of truth is: {(res[1]/4)*100}"
    percentage = res[2]
    pred_LR = percentage[0]
    pred_DT = percentage[1]
    pred_GBC = percentage[2]
    pred_RFC = percentage[3]

    eval_res = ((0.4*pred_LR) + (0.1*pred_DT) + (0.1*pred_GBC) + (0.4*pred_RFC)) * 100
    
    result_text = f"{res[0]}\nScore is: {res[1]}/4,\n Percentage of truth is: {eval_res} \nVerdict: {verdict(res[1])}" # (res[1]/4)*100
    result_lines = result_text.split('\n')
    result_with_br = '<br>'.join(result_lines)
    return result_with_br

def detect_language(text):
    try:
        lang_info = detect_langs(text)
        if lang_info:
            return lang_info[0].lang
        else:
            return "Unknown"
    except:
        return "Unknown"

@app.route('/')
def index():
    return render_template('index.html', translation="", result="", input_lang="")

@app.route('/', methods=['POST'])
def process():
    news = request.form['news']
    input_lang = detect_language(news)
    translated_news = translate_to_english(news)
    res = ResultsEvaluation(translated_news)
    result = PrintResults(res)
    return render_template('index.html', translation=translated_news, result=result, input_lang=input_lang)



if __name__ == '__main__':
    app.run(debug=True)
