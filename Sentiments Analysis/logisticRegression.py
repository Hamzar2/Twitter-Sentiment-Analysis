import findspark
import pyspark
import time
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StringIndexer, CountVectorizer, NGram, VectorAssembler, ChiSqSelector
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression , LinearSVC , DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from datetime import datetime
from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame 
from pyspark.ml.feature import StopWordsRemover


findspark.init()
spark1 = SparkSession.builder\
            .master("local[3]")\
            .appName("LR_Model")\
            .config("spark.executor.memory", "6g")\
            .getOrCreate() 

path = "twitter_cleaned_data.csv"
    
schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("source", StringType(), True),
    StructField("target", StringType(), True),
    StructField("tweet", StringType(), True)])

df = spark1.read.csv(path,
                    inferSchema=True,
                    header=False,
                    schema=schema)

df = df.filter(df.tweet.isNotNull())
df.show()

(train_set, test_set) = df.randomSplit([0.8, 0.2])

tokenizer = Tokenizer(inputCol="tweet", outputCol="tk")
stopwordsRemover = StopWordsRemover(inputCol="tk", outputCol="filtered_words")
ngram = NGram(inputCol="filtered_words", outputCol="ngrams", n=2) 
hashtf = CountVectorizer(inputCol="ngrams", outputCol='tf')
idf = IDF(inputCol='tf', outputCol="features")
label_stringIdx = StringIndexer(inputCol = "target", outputCol = "label")
lr = LogisticRegression()

pipeline = Pipeline(stages=[ tokenizer,stopwordsRemover , ngram, hashtf, idf, label_stringIdx, lr])

print("Training ........")
pipelineFit = pipeline.fit(train_set)

#print("Saving the model ........")
#pipelineFit.save("LogisticRegressionModel")

predictions = pipelineFit.transform(test_set)
predictions = predictions.select(predictions.tweet , predictions.label , predictions.prediction)
predictions.show()

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

print("Accuracy:", accuracy)
print("f1:", f1)


