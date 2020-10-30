import time #
start_time = time.time()
import os
import re
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import  pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import ktrain
from ktrain import text
from matplotlib import pyplot
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.utils import class_weight
from collections import Counter


def configuration():
    language = 'Arabic'
    # file_name = 'en_train_2_data_from_all_years'
    training_path = r'C:\Users\user\Google Drive\סאיקאן\Projects\Shivat Zion\Arabic\Training\antisemite training 3-new as doubled-no legitimate critics-more 150 not as.xlsx'

    file_name = training_path.split("\\")[-1][:-5]
    test_path = r'C:\Users\user\Google Drive\סאיקאן\Projects\Shivat Zion\Arabic\Test\ronen-test-22-10-20-ordered.xlsx'
    save_path = r'.\Models'
    test_size = 0.2
    threshold = 0.6
    global epochs
    epochs = 1
    return training_path, save_path, test_path, test_size, threshold

training_path, save_path, test_path, test_size, threshold = configuration()


def clean_text(text):
    eng_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                   'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                   'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    # english_free = ''.join([ch for ch in doc if ch not in eng_letters])
    # stop_free = ' '.join([i for i in punc_free.split() if i not in arb_long_sw])
    try:
        stop_free = re.sub("^RT:", "", text)
        stop_free = re.sub("User Location:.*", "", stop_free)
        stop_free = re.sub(r'http\S+', "", stop_free)
        stop_free = re.sub("@.*:", "", stop_free)
        # stop_free = re.sub("@.*", "", stop_free)
        stop_free = stop_free.strip()
    except:
        stop_free = ''
        print('cant parse the text')

    return stop_free


def predict_test_threshold(text, predictor):
    text_predict = [predictor.predict(text)][0]
    text_predict_prob = np.max(predictor.predict_proba(text), axis=1)
    # print(np.max(predictor.predict_proba(text), axis=1))
    # for i in range(len(text_predict)):
    #     if (str(text_predict[i]) != '0.0'):
    #         if (text_predict_prob[i] < threshold):
    #             text_predict[i] = '0.0'
    text_predict = [str(y) for y in text_predict]
    # text_predict_prob = np.max(predictor.predict_proba(text), axis=1)

    return text_predict  # , text_predict_prob

def print_results():
    labels_sort = t.get_classes()
    # labels_for_confusion_mat['1.06.01'] = '1.0601'
    # labels_for_confusion_mat = labels_for_confusion_mat.sort(key=str)
    # print((t.get_classes()).sort(key=str))

    print('training classification report:')
    classification_report_print(x_train, y_train, labels_sort)


    print('\nvalidation classification report:')
    classification_report_print(x_val, y_val)
    # print('\ntest classification report:')
    # classification_report_print(x_test, y_test)
    # confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])

def explain():
    predictor = ktrain.load_predictor(r'C:\Users\srulik\Heavy Files\Russian\Training 13-srul version')
    print(predictor.predict('RT: @RaphJackAP: En ce moment, on dit pas youpin mais youmatsa.Mettez vous à jour. https://t.co/54YmANTDII User Location:Paris, France'))
    predictor.explain('Jesus Christ is the central figure of Christianity.')
    pyplot.show()

def class_weights(class_names, df):
    class_names = [float(cls) for cls in class_names]
    class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                                     classes=np.array(class_names),
                                                                     y=np.array(df['label'].tolist()))))
    print('class_weights: ' + str(class_weights))
    return class_weights


def standard_classification_pipeline(df):
    # multilabel_index(df)
    # print('multilabel')
    # text_dic = {}
    # df = df.sort_values(by='post')
    # df_unduplicate = df.drop_duplicates(subset=['post'], inplace=False)
    # df_unduplicate = df_unduplicate.sort_values(by='post')
    # unique_text = df_unduplicate['post'].iloc[0]
    # labels_aggragate_sets=[]
    # labels_set = set()
    # cou=0
    # for i in range(len(df)):
    #     # print(i)
    #     if(df['post'].iloc[i]==unique_text):
    #         labels_set.add(str(df['label'].iloc[i]))
    #     else:
    #         labels_aggragate_sets.append(labels_set)
    #         labels_set = set()
    #         labels_set.add(str(df['label'].iloc[i]))
    #         cou+=1
    #         unique_text = df_unduplicate['post'].iloc[cou]
    # labels_aggragate_sets.append(labels_set)
    #
    # # return labels_aggragate_sets
    # mlb = MultiLabelBinarizer()
    # # labels_set=[set(i) for i in labels_set]
    #
    # multilabel_transform = mlb.fit_transform(labels_aggragate_sets)

    # df_post_clean = [clean_text(doc) for doc in df_unduplicate['post']]
    # x_train, x_val, y_train, y_val = train_test_split(df_post_clean, multilabel_transform, test_size=0.1 ,shuffle=True)    #random_state = np.random.randint(1,1000))
    # class_names = list(set(df_unduplicate['label'].tolist()))

    # df = df_concat

    # df_sampled = training_sampling(df, df['label'].drop_duplicates().tolist(), [250,250,250,250,250,250,250,250,250,250,250,250,250,250])
    # df = df[df['label'] != 'nan']
    # counter(df)
    # df_sampled = df
    # df_sampled = df

    # df_post_clean = [doc for doc in df_sampled['post']]
    df_post_clean = [clean_text(doc) for doc in df['tweet']]
    # df_post_clean = [clean_text(doc) for doc in df_unduplicate['post']]

    df_post_clean_to_excel = pd.DataFrame()
    df_post_clean_to_excel['tweet'] = df_post_clean
    df_post_clean_to_excel['label'] = df['label']
    # df_post_clean_to_excel.drop_duplicates(subset=['post clean'], inplace=True)
    df_post_clean = df_post_clean_to_excel['tweet'].tolist()
    # df_post_clean_to_excel.to_excel('post clean.xlsx')

    # x_train = df['post'].tolist()
    # y_train = df['label'].tolist()
    # x_test = df_test['tweet'].tolist()
    # y_test = df_test['label'].tolist()
    # y_test = [str(y) for y in y_test]

    # x_train, x_val, y_train, y_val = train_test_split(df_post_clean, [labels_dic[label] for label in df_sampled['label'].tolist()], test_size=0.2, shuffle=False)
    # df_label_dic = [labels_dic[y] for y in df['label'].tolist()]
    # for i in range(len(df['label'][-151:])):
    #     print(i)
    #     print(labels_dic[str(df_label_list[i])])
    # x_train, x_val, y_train, y_val = train_test_split(df_post_clean, [str(labels_dic[y]) for y in df['label'].tolist()], test_size=10, shuffle=True)    #random_state = np.random.randint(1,1000))

    # classes_np =  np.array(df_post_clean_to_excel['label'].tolist()).astype(np.float)

    x_train, x_val, y_train, y_val = train_test_split(df_post_clean,
                                                      [str(y) for y in df_post_clean_to_excel['label'].tolist()],
                                                      test_size=test_size,
                                                      shuffle=True)  # random_state = np.random.randint(1,1000))

    # y_train = np.asarray(y_train, dtype=np.float32)
    # y_val = np.asarray(y_val, dtype=np.float32)
    # y_test = df_test['label'].tolist()
    # y_test = np.asarray(y_test, dtype=np.float32)
    # x_test = df_test['post'].tolist()

    # class_names = [label for label in list(set(y_train))]
    # class_names = list(set(y_train))
    MODEL_NAME = 'distilbert-base-multilingual-cased' #distilled version for faster ttraining and inference
    # MODEL_NAME = 'bert-base-multilingual-cased' #for better results

    # class_names = list(set(df['label'].tolist()))
    # class_names = [str(cls) for cls in class_names]
    # class_names = list(set(class_names))
    # class_names.sort()
    t = text.Transformer(MODEL_NAME, maxlen=50)#, class_names=class_names)

    # classes = t.get_classes()
    # classes = [str(cls) for cls in classes]
    # class_weight = class_weights(class_names, df)
    trn = t.preprocess_train(x_train, y_train)
    val = t.preprocess_test(x_val, y_val)
    model = t.get_classifier()
    print(t.get_classes())
    learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=8)
    # learner.fit_onecycle(5e-5, epochs=epochs, class_weight=class_weight)
    # learner.fit_onecycle(5e-5, epochs=epochs)
    predictor = ktrain.load_predictor(r'.\Models')

    # print('train number of class 1: ' + str(y_train.sum()))
    # print('train number of class 0: ' + str(len(y_train)-y_train.sum()))
    # learner.lr_find(max_epochs=1, start_lr=1e-6, show_plot=True) # briefly simulate training to find good learning rate
    # learner.lr_plot()               # visually identify best learning rate
    # pyplot.show()

    # print_results()

    # predictor = ktrain.get_predictor(learner.model, preproc=t)
    # predictor.save(save_path)


    # print('training classification report:')
    # learner.validate()
    return predictor
    # classification_report_print(x_train, y_train, 'training', class_names, predictor, labels_dic_names, t)
    # print('\nvalidation classification report:')
    # classification_report_print(x_val, y_val, 'validation', class_names, predictor, labels_dic_names, t)
    # print('\ntest classification report:')
    # classification_report_print(x_test, y_test, 'test', class_names, predictor, labels_dic_names, t)



