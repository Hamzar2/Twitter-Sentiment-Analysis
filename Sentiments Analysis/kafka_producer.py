from kafka import KafkaProducer
import atexit
import logging
import argparse
import json
import pandas as pd
import numpy as np
import time
import re

logging.basicConfig(level=logging.INFO)

search_term = 'ChatGPT'
topic_name = 'twitter'
logger = logging.getLogger()


class ProducerCallback:
    def __init__(self, record, path_data, log_success=False):
        self.record = record
        self.log_success = log_success
        self.path_data = path_data

    def __call__(self, err, msg):
        if err:
            logger.error('Error streaming record {}'.format(self.record))
        elif self.log_success:
            logger.info('tweet {} to topic {} partition {} offset {}'.format(
                self.record,
                msg.topic(),
                msg.partition(),
                msg.offset()
            ))


def main(args , path_data):
    logger.info('Starting the producer')

    producer = KafkaProducer(bootstrap_servers=args.bootstrap_server)

    atexit.register(lambda p: p.flush(), producer)

    #csv_file_path = r"flask-datta-able\Sentiments Analysis\twitter_validation.csv"
    df = pd.read_csv(path_data , header=None)
    i = 1
    print(df)
    while True:
        index = np.random.randint(0,900)
        #print("a" , df.iloc[index-1 : index , 3].to_string())

        msg = df.iloc[index-1 : index , 3].to_string()
        result = re.sub(r'[^A-Za-z\n ]|(http\S+)|(www.\S+)', '', msg.lower().strip()).split()
        cleaned_text = ' '.join(result)
        
        data = {
            'id' : index,
            'source' : df.iloc[index-1 : index , 1].to_string().split()[1],
            'target' : df.iloc[index-1 : index , 2].to_string().split()[1],
            'tweet': cleaned_text.replace(',', '') 
        }
        print(data)
        producer.send(topic_name, value=json.dumps(data).encode('utf-8'))
        time.sleep(10)
        i+=1

if __name__ == "__main__":
    path_data = r"C:\Users\DR2\Desktop\IASD\S2\BigData\Twitter Sentiment Analysis\Sentiments Analysis\twitter_validation.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap-server', default='localhost:9092')
    parser.add_argument('--topic', default="twitter")
    args = parser.parse_args()
    main(args , path_data)