from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = FastAPI()

# Load models and vectorization
LogReg_model = joblib.load('LogisticRegression.pkl')
DesTree_model = joblib.load("DecisionTree.pkl")
GradBoostingClassifier_model = joblib.load('GradientBoostingClassifier.pkl')
RandForClassifier_model = joblib.load('RandomForestClassifier.pkl')
vectorization = joblib.load('vectorization.pkl')

class NewsItem(BaseModel):
    news: str

def better_text_processing(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|<.*?>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
    return ' '.join(tokens)

@app.post("/predict")
async def predict(news_item: NewsItem):
    news_testing = {"text": [news_item.news]}
    processed_news = better_text_processing(news_item.news)
    sam_xv_test = vectorization.transform([processed_news])
    pred_RFC = RandForClassifier_model.predict(sam_xv_test)[0]
    pred_LR = LogReg_model.predict(sam_xv_test)[0]
    pred_GBC = GradBoostingClassifier_model.predict(sam_xv_test)[0]
    pred_DT = DesTree_model.predict(sam_xv_test)[0]
    score = [pred_LR, pred_DT, pred_GBC, pred_RFC].count(1)
    return {
        "LR Prediction": pred_LR,
        "DT Prediction": pred_DT,
        "GBC Prediction": pred_GBC,
        "RFC Prediction": pred_RFC,
        "score": score
    }
