# -- encoding: utf-8 --
"""
Copyright (c) 2019 - present AppSeed.us
"""

import os
from   flask_migrate import Migrate
from   flask_minify  import Minify
from   sys import exit
from flask import Flask, render_template , jsonify
import csv
from apps.config import config_dict
from apps import create_app, db
from collections import Counter
import pandas as pd
from flask import Flask, render_template
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re , sys
import findspark
import logging
import argparse
import threading
from time import sleep
from pymongo import MongoClient


Predictions = []
acc = 0
f1 = 0
sys.path.append(os.path.abspath('./Sentiments Analysis'))

from  kafka_consumer import TwitterSentimentAnalyzer
from  kafka_producer import ProducerCallback , main

# WARNING: Don't run with debug turned on in production!
DEBUG = (os.getenv('DEBUG', 'False') == 'True')

# The configuration
get_config_mode = 'Debug' if DEBUG else 'Production'

try:

    # Load the configuration using the default values
    app_config = config_dict[get_config_mode.capitalize()]

except KeyError:
    exit('Error: Invalid <config_mode>. Expected values [Debug, Production] ')

app = create_app(app_config)
Migrate(app, db)

if not DEBUG:
    Minify(app=app, html=True, js=False, cssless=False)

if DEBUG:
    app.logger.info('DEBUG            = ' + str(DEBUG)             )
    app.logger.info('Page Compression = ' + 'FALSE' if DEBUG else 'TRUE' )
    app.logger.info('DBMS             = ' + app_config.SQLALCHEMY_DATABASE_URI)
    app.logger.info('ASSETS_ROOT      = ' + app_config.ASSETS_ROOT )

def load_data_from_csv(csv_file):
    data = []
    with open(csv_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # skip the header row
        for row in csv_reader:
            tweet_id, entity, sentiment, tweet_content = row
            data.append({
                'Tweet ID': int(tweet_id),
                'Entity': entity,
                'Sentiment': sentiment,
                'Tweet content': tweet_content
            })
    return data

def count_sentiments_by_entity(data):
    sentiments_by_entity = {}
    for row in data:
        entity = row['Entity']
        sentiment = row['Sentiment']
        if entity not in sentiments_by_entity:
            sentiments_by_entity[entity] = {}
        if sentiment not in sentiments_by_entity[entity]:
            sentiments_by_entity[entity][sentiment] = 0
        sentiments_by_entity[entity][sentiment] += 1
    return sentiments_by_entity


def calculate_sentiment_percentages():
    data = load_data_from_csv(r'apps\twitter_training - twitter_training.csv')
    sentiment_counts = Counter(row['Sentiment'] for row in data)
    total = sum(sentiment_counts.values())
    percentages = {sentiment: (count / total) * 100 for sentiment, count in sentiment_counts.items()}
    return percentages

def get_top_words_by_sentiment(data, top_n=10):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    sentiment_words = {'positive': [], 'negative': [], 'neutral': []}
    word_pattern = re.compile(r'^[a-zA-Z]+$')  # Regex pour ne garder que les mots avec des lettres

    for row in data:
        sentiment = row['Sentiment'].lower()  # Normalisez les sentiments en minuscules
        if sentiment not in sentiment_words:
            continue  # Ignorer les sentiments non pris en compte
        tweet_content = row['Tweet content'].lower()
        words = word_tokenize(tweet_content)
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word_pattern.match(word)]
        sentiment_words[sentiment].extend(words)

    top_words_by_sentiment = {}
    for sentiment, words in sentiment_words.items():
        word_counts = Counter(words)
        top_words_by_sentiment[sentiment] = word_counts.most_common(top_n)

    return top_words_by_sentiment


@app.route('/chart-morris.html')
def index():
    csv_file = r'apps\twitter_training - twitter_training.csv' # Ajustez ceci au chemin de votre fichier CSV
    data = load_data_from_csv(csv_file)
    sentiments_by_entity = count_sentiments_by_entity(data)
    sentiment_percentages = calculate_sentiment_percentages()
    top_words_by_sentiment = get_top_words_by_sentiment(data)
    return render_template('home/chart-morris.html', sentiments_by_entity=sentiments_by_entity, sentiment_percentages=sentiment_percentages, top_words_by_sentiment=top_words_by_sentiment)

@app.route('/tbl_bootstrap.html') 
def table():
    data = load_data_from_csv(r'apps\twitter_training - twitter_training.csv')  # Mettez le nom de votre fichier CSV ici
    return render_template('home/tbl_bootstrap.html', data=data[:100])

def consumer():
    findspark.init()
    path_data = r"C:\Users\DR2\Desktop\IASD\S2\BigData\Twitter Sentiment Analysis\Sentiments Analysis\twitter_cleaned_data.csv"
    global analyzer 
    analyzer = TwitterSentimentAnalyzer(path_data)
    analyzer.create_spark_session()
    analyzer.create_pipeline()
    analyzer.model()
    

MONGO_URI = 'mongodb+srv://twitter:1234@cluster0.u0w2sya.mongodb.net/'
MONGO_DB = 'twitter'
client = MongoClient(MONGO_URI)
db = client[MONGO_DB]

@app.route('/api/data')
def get_data():
    
    collection = db.predictions
    data = list(collection.find())
    #print(data)
    processed_data = []
    for doc in reversed(data):  # Reverse the order of data
        processed_data.append({
            'tweet': doc.get('tweet', ''),
            'target': doc.get('target', ''),
            'label': doc.get('label', ''),
            'filtered_words': doc.get('filtered_words', ''),
            'prediction': doc.get('prediction', '')
        })
    return jsonify(processed_data)

@app.route("/get_predictions")
def get_predictions():
    """Fetches accuracy and F1 score from the 'model' collection and returns as JSON."""
    collection = db.model  
    
    # Fetch the latest document (assuming you store the latest metrics)
    latest_metrics = list(collection.find())
    for doc in reversed(latest_metrics):
        processed_data = {
            'accuracy': doc.get('accuracy', 0.0),
            'f1': doc.get('f1', 0.0)          
        }
        print(processed_data)
        return jsonify(processed_data)

def streaming():
    path_data = r"Sentiments Analysis\twitter_validation.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap-server', default='localhost:9092')
    parser.add_argument('--topic', default="twitter")
    args = parser.parse_args()
    main(args , path_data)


def dashboard():
    app.run(debug=DEBUG, use_reloader=False)

def test():
    response_data = {
        "predictions": analyzer.modelPredictions,
        "accuracy": analyzer.acc,
        "f1_score": analyzer.f1
    }
    return response_data

if __name__ == "__main__":
    #consumer()
    app.run(debug=DEBUG, use_reloader=False)
    