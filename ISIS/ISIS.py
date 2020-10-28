# optional features: name, ssimilar names, day of week of the tweets, number of radical tweets, percentage of radical tweets
import sys
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV
from BERT import standard_classification_pipeline



def configuration():
    pass


def load_data():
    isis_data = pd.read_excel(r'.\tweets_isis_all.xlsx')
    random_data = pd.read_excel(r'.\tweets_random_all.xlsx')
    tweets_only_labeled = pd.read_excel(r'.\tweets_only_labeled.xlsx')
    return tweets_only_labeled, isis_data, random_data

def preprocessing():
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



def train_classifiers():
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

def inference_pipeline():
    pass


def twitter(twitter_id):
    import tweepy
    user_tweets=[]
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    public_tweets = api.home_timeline()
    #for tweet in public_tweets:
        #user_tweets.append(tweet.text)
    user_tweets = [tweet.text for tweet in public_tweets]
        
    user = api.get_user('twitter')
    user_screen_name = user.screen_name
    user_followers_count = user.followers_count
    #for friend in user.friends():
       #print(friend.screen_name)
    user_friends_screen_name = [friend.screen_name for friend in user.friends()]
    
    return 



if __name__ == '__main__':
    # if ('train' in sys.argv):
    train_pipeline()
    # else:
    #     inference_pipeline()




