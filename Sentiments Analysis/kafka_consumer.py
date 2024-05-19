import re
import findspark
import os
import sys
import pandas as pd
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline ,PipelineModel
from pyspark.ml.feature import Tokenizer , StopWordsRemover , CountVectorizer , IDF , NGram , ChiSqSelector ,StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, udf, col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType , DoubleType
from pymongo import MongoClient
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

class TwitterSentimentAnalyzer:
    def __init__(self, path_data, bootstrap_servers="localhost:9092", topic="twitter" , mongo_db="twitter"):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.spark = None
        self.loadedLRModel = None
        self.pipeline = None
        self.path_data = path_data
        self.mongo_uri = "mongodb+srv://twitter:1234@cluster0.u0w2sya.mongodb.net/"
        self.mongo_db = mongo_db
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.mongo_db]
        self.collection = "predictions"

    def create_spark_session(self):
        
        self.spark = SparkSession \
            .builder \
            .master("local[*]") \
            .appName("TwitterSentimentAnalysis") \
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel('ERROR')

    def create_pipeline(self):
        tokenizer = Tokenizer(inputCol="tweet", outputCol="tk")
        stopwordsRemover = StopWordsRemover(inputCol="tk", outputCol="filtered_words")
        ngram = NGram(inputCol="filtered_words", outputCol="ngrams", n=2) 
        hashtf = CountVectorizer(inputCol="tk", outputCol='tf')
        idf = IDF(inputCol='tf', outputCol="features")
        label_stringIdx = StringIndexer(inputCol = "target", outputCol = "label")
        lr = LogisticRegression()
        self.pipeline = Pipeline(stages=[ tokenizer,stopwordsRemover ,ngram, hashtf, idf, label_stringIdx, lr])
        print("Pipeline Created Successfully.")

    def process_batch(self, batch_df):
            if batch_df.isEmpty():
                print("Empty batch received. Skipping prediction.")
                return
            batch_df.printSchema()
            try:
                """pipelineModel = self.pipeline.fit(batch_df)
                transformed_data = pipelineModel.transform(batch_df)
                if 'prediction' in transformed_data.columns:
                    transformed_data = transformed_data.drop('prediction')"""
                prediction = self.loadedLRModel.transform(batch_df)
                prediction = prediction.select(prediction.tweet ,prediction.filtered_words, prediction.target , prediction.label , prediction.prediction)
                prediction.show()
                prediction_pd = prediction.toPandas()
                prediction_dict = prediction_pd.to_dict("records")

                collection = self.db[self.collection]
                collection.insert_many(prediction_dict)

            except Exception as e:
                print(f"Error in process_batch: {e}")
                batch_df.show()
                raise e

    def start_streaming(self):
        
        schema = StructType([
            StructField("id", IntegerType(), True), 
            StructField("source", StringType(), True), 
            StructField("target", StringType(), True), 
            StructField("tweet", StringType(), True) 
        ])

        df = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.bootstrap_servers) \
            .option("subscribe", self.topic) \
            .load()

        df = df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), schema).alias("data")) \
            .select(
                col("data.id").alias("id"),
                col("data.source").alias("source"),
                col("data.target").alias("target"),
                col("data.tweet").alias("tweet"),
            )

        query = df \
            .writeStream \
            .format("console") \
            .outputMode("update") \
            .foreachBatch(lambda batch_df, batch_id: self.process_batch(batch_df)) \
            .start()

        query.awaitTermination()
    
    def model(self):
        schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("source", StringType(), True),
            StructField("target", StringType(), True),
            StructField("tweet", StringType(), True)])
        
        df = self.spark.read.csv(self.path_data,
                            inferSchema=True,
                            header=False,
                            schema=schema).sample(0.5)

        df = df.filter(df.tweet.isNotNull())
        df.show()

        (train_set, test_set) = df.randomSplit([0.8, 0.2])
        self.loadedLRModel = self.pipeline.fit(train_set)
        print("Training done.")

        prediction = self.loadedLRModel.transform(test_set)
        prediction = prediction.select(prediction.tweet , prediction.filtered_words, prediction.label , prediction.prediction)

        evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
        accuracy = evaluator.evaluate(prediction, {evaluator.metricName: "accuracy"})
        f11 = evaluator.evaluate(prediction, {evaluator.metricName: "f1"})

    
        self.modelPredictions = []
        self.acc = accuracy
        self.f1 = f11

        print("Accuracy:", accuracy)
        print("f1:", f11)

        
        prediction_pd = prediction.toPandas()
        prediction_dict = prediction_pd.to_dict("records")
        self.modelPredictions.extend(prediction_dict)

