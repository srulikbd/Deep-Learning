

def location_in_ISIS_loc(known_location_set, loc):
    if(loc in known_location_set):
        return 1
    else:
        return 0

def max_day_of_week_tweets(tweets_timestamps):
    pass

def num_isis_tweets_threshold_norm(tweets, tweets_prob, threshold):
    cou=0
    for i in range(len(tweets)):
        if(tweets_prob[i]>=threshold):
            cou+=1

    return cou/tweets


def description_in_ISIS_data(description, ISIS_description_dict):

    if(description in ISIS_description_dict):
        return 1
    else:
        return 0

def mention_ISIS_user(ISIS_users_dict, mentions, num_tweets):
    cou=0
    flag=0
    for i in range(len(mentions)):
        if(mentions[i] in ISIS_users_dict):
            cou+=1
    if(cou>0):
        flag=1
    return flag, cou, cou/num_tweets


def build_user_feature_matrix(tweets):
    user_feature_matrix=[]
    tweet_prob = predictor.predict

    user_feature_matrix.append(num_isis_tweets_threshold_norm(tweets, tweets_prob, threshold))
    # user_feature_matrix.append(location_in_ISIS_loc())
    # user_feature_matrix.append(max_day_of_week_tweets())
    # user_feature_matrix.append(description_in_ISIS_data())
    # user_feature_matrix.append(mention_ISIS_user())

    return  user_feature_matrix



