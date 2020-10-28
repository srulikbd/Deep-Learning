# optional features: name, ssimilar names, day of week of the tweets, number of radical tweets, percentage of radical tweets
import pandas as pd
from sklearn import descisicon_trre, random forest
from BERT import *


def configuration():



def load_data():
    isis_data = pd.load_excel(r'C:\Users\user\srulik\AF\tweets_isis_all.xlsx')
    random_data = pd.load_excel(r'C:\Users\user\srulik\AF\tweets_random_all.xlsx')

    return isis_data, random_data

def preprocessing():


def features_creation(isis_data):
    exact_location = isis_data['location'].tolist()

    num_ISIS_followers





def train_BERT(isis_data, random_Data):
    standard_classification_pipeline(isis_data['tweets'], random_Data['content'].tolist())


def train():


def validation():









if __name__ == '__main__':
    isis_data, random_data = load_data()


