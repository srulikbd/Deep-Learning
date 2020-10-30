# optional features: name, ssimilar names, day of week of the tweets, number of radical tweets, percentage of radical tweets
import sys
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV
from BERT import standard_classification_pipeline, predict_test_threshold
from features import build_user_feature_matrix



def configuration():
    predictor = ktrain.load_predictor(r'C:\Users\user\srulik\Heavy Files\Shivat_Zion_Models\Arabic\antisemite training 3-new as doubled-no legitimate critics')

    return predictor


def load_data():
    isis_data = pd.read_excel(r'.\tweets_isis_all.xlsx')
    random_data = pd.read_excel(r'.\tweets_random_all.xlsx')
    tweets_only_labeled = pd.read_excel(r'.\tweets_only_labeled_small_sample.xlsx')
    tweets_only_labeled = pd.read_excel(r'.\tweets_labeled_ISIS_keyword.xlsx')
    return tweets_only_labeled, isis_data, random_data

def aggragate_user_tweets(users_name, tweets): # the data is already orderd by username in the excel file
    users_tweets = []
    current_user_tweets=[]
    current_user = users_name[0]

    for i in range(len(tweets)):
        current_user_tweets.append(tweets[i])
        if(current_user!=users_name[i+1]):
            users_tweets.append(current_user_tweets)
            current_user_tweets=[]
            current_user = users_name[i+1]

    return users_tweets





    pass

def features_creation(isis_data):
    exact_location = isis_data['location'].tolist()

    num_ISIS_followers


def creat_feature_vectors():
    pass



def train_BERT_tweet(isis_data, random_Data):
    standard_classification_pipeline(isis_data['tweets'], random_Data['content'].tolist())

def train_BERT_description(isis_data, random_Data):
    standard_classification_pipeline(isis_data['description'], random_Data['content'].tolist())



def train_classifiers(X, y):
    clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2)
    scores = cross_val_score(clf, X, y, cv=5)
    DecisionTreeClassifier_mean_scores = scores.mean()

    clf = RandomForestClassifier(n_estimators=10, max_depth=None)
    scores = cross_val_score(clf, X, y, cv=5)
    RandomForestClassifier_mean_scores = scores.mean()

    clf = ExtraTreesClassifier(n_estimators=10, max_depth=None)
    scores = cross_val_score(clf, X, y, cv=5)
    ExtraTreesClassifier_mean_scores = scores.mean()

    X, y = load_iris(return_X_y=True)
    clf = AdaBoostClassifier(n_estimators=100)
    scores = cross_val_score(clf, X, y, cv=5)
    AdaBoostClassifier_mean_scores = scores.mean()

    X, y = load_iris(return_X_y=True)
    clf = LogisticRegressionCV(cv=5, random_state=0).fit(X, y)
    cv = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=np.random.randint(1, 1000))
    cv_score = cross_val_score(clf, X, y, cv=cv, scoring='precision')
    cv_score_average = cv_score.mean()

def choose_best_classifier():
    pass

def validation():
    pass

def predict():
    pass



def train_pipeline():
    tweets_only_labeled, isis_data, random_data = load_data()
    standard_classification_pipeline(tweets_only_labeled)
    users_feature_mat=[]
    users_names = isis_data['username'].tolist()
    users_tweets = aggragate_user_tweets(users_names, isis_data['tweets'].tolist())
    predict_test_threshold(tweets_only_labeled, predictor)
    for i in range(len(users_names)):
        user_feature_vector = build_user_feature_matrix()
        users_feature_mat.append(user_feature_vector)

    train_classifiers(users_feature_mat, users_labels)

def inference_pipeline():
    pass




if __name__ == '__main__':
    # if ('train' in sys.argv):
    predictor = configuration()
    train_pipeline()
    inference_pipeline()
    # else:
    #     inference_pipeline()




