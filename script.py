
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
# from sklearn.cross_validation import cross_val_score
import logging
import xgboost as xgb

pd.options.mode.chained_assignment = None  # default='warn'

# Classification inside one grid cell.
def process_one_cell(df_cell_train, df_cell_test, fw, th, crossValidation, k):


    #Working on df_train
    # place_counts = df_cell_train.place_id.value_counts()
    # mask = (place_counts[df_cell_train.place_id.values] >= th).values
    # df_cell_train = df_cell_train.loc[mask]
    score = 0
    #Working on df_test
    row_ids = df_cell_test.index
    #Feature engineering on x and y
    df_cell_train.loc[:,'x'] *= fw[0]
    df_cell_train.loc[:,'y'] *= fw[1]
    df_cell_test.loc[:,'x'] *= fw[0]
    df_cell_test.loc[:,'y'] *= fw[1]
    # train, validate = train_test_split(df_cell_train, test_size = 0.1)
    train = df_cell_train
    place_counts = train.place_id.value_counts()
    mask = (place_counts[train.place_id.values] >= th).values
    df_cell_train = train.loc[mask]

    #Preparing data
    le = LabelEncoder()
    # le.fit(df_cell_train.place_id.values)
    y = le.fit_transform(df_cell_train.place_id.values)
    X = df_cell_train.drop(['place_id'], axis=1).values
    # get the validation data
    # X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2,random_state=0)
    X_test = df_cell_test.values

    # construct DMatrices
    dm_train = xgb.DMatrix(X,label = y)
    dm_test = xgb.DMatrix(X_test)
    #Applying the classifier
    # clf = KNeighborsClassifier(n_neighbors=k, weights='distance',metric='manhattan', n_jobs = -1)
    # clf.fit(X, y)

    booster = xgb.train(
            {'eta': 0.1, 'objective': 'multi:softprob',
             'num_class': len(le.classes_),
             'alpha': 0.1, 'lambda': 0.1, 'booster': 'gbtree'},
            dm_train, num_boost_round=10, verbose_eval=10)
    print(booster)
    # predict_y_train = booster.predict(dm_train)
    predict_y_test = booster.predict(dm_test)
    # clf.fit(X_train, y_train)
    # if crossValidation == True:
    #     # prepare the validate data
    #     y_validate = le.fit_transform(validate.place_id.values)
    #     X_validate = validate.drop(['place_id'], axis=1).values
    #     # get the scores of cross validation
    #     # scores = cross_val_score(clf, X, y, cv=5, n_jobs=1)
    #     y_pred_validation = clf.predict(X_validate)
    #     # score = clf.score(X_validate,y_validate)
    #     score = precision_score(y_validate, y_pred_validation, average = 'weighted')
    #     # logging.info(score)
    #     # logging.info(scores.std())
    # y_pred = clf.predict_proba(X_test)
    # print ('aaa%s ',y_pred.shape)

    pred_labels = le.inverse_transform(np.argsort(predict_y_test, axis=1)[:,::-1][:,:3])

    return pred_labels, row_ids,score

def process_grid(df_train, df_test, size, x_step, y_step, x_border_augment, y_border_augment, fw, th, k):
    """
    Iterates over all grid cells, aggregates the results and makes the
    submission.
    """
    scores = []
    preds = np.zeros((df_test.shape[0], 3), dtype=int)

    for i in range((int)(size/x_step)):
        start_time_row = time.time()
        x_min = x_step * i
        x_max = x_step * (i+1)
        x_min = round(x_min, 4)
        x_max = round(x_max, 4)
        if x_max == size:
            x_max = x_max + 0.001

        df_col_train = df_train[(df_train['x'] >= x_min-x_border_augment) & (df_train['x'] < x_max+x_border_augment)]
        df_col_test = df_test[(df_test['x'] >= x_min) & (df_test['x'] < x_max)]
        for j in range((int)(size/y_step)):
            crossValidation = False
            y_min = y_step * j
            y_max = y_step * (j+1)
            y_min = round(y_min, 4)
            y_max = round(y_max, 4)
            if y_max == size:
                y_max = y_max + 0.001
            if j == 1 and i == 1:
                crossValidation = True
            # elif j == 5 and i == 5:
            #     crossValidation = True
            # elif i == 10:
            #     crossValidation = True
            # elif j == 11 and i == 11:
            #     crossValidation = True
            # elif j == 15 and i == 15:
            #     crossValidation = True
            # elif j == 2 and i == 2:
            #     crossValidation = True
            # elif j == 3 and i == 3:
            #     crossValidation =True
            # elif j == 4 and i == 4:
            #     crossValidation = True
            else:
                continue

            # if j == 4:
            #     crossValidation = True
            # if j == 5:
            #     crossValidation = True
            logging.info(i,j)
            df_cell_train = df_col_train[(df_col_train['y'] >= y_min-y_border_augment) & (df_col_train['y'] < y_max+y_border_augment)]
            df_cell_test = df_col_test[(df_col_test['y'] >= y_min) & (df_col_test['y'] < y_max)]

            #Applying classifier to one grid cell
            pred_labels, row_ids, score = process_one_cell(df_cell_train, df_cell_test, fw, th, crossValidation, k)
            if score != 0:
                scores.append(score)
            #Updating predictions
            preds[row_ids] = pred_labels

        logging.info("Row %d/%d elapsed time: %.2f seconds" % (i+1, (int)(size/x_step),(time.time() - start_time_row)))

    logging.info('Generating submission file ...')

    df_aux = pd.DataFrame(preds, dtype=str, columns=['l1', 'l2', 'l3'])

    #Concatenating the 3 predictions for each sample
    ds_sub = df_aux.l1.str.cat([df_aux.l2, df_aux.l3], sep=' ')

    #Writting to csv
    ds_sub.name = 'place_id'
    logging.info(np.mean(scores))
    # ds_sub.to_csv('sub_knn.csv', index=True, header=True, index_label='row_id')


##########################################################
# Main
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='myExperiments1.txt',
                        filemode='w')
    # Input varialbles
    fw = [500., 1000., 4., 3., 2., 10., 10.] #feature weights (black magic here)
    th = 5 #Keeping place_ids with more than th samples.

    #Defining the size of the grid
    size = 10.0
    x_step = 0.2
    y_step = 0.2
    k = 25
    x_border_augment = 0.025
    y_border_augment = 0.0125
    # logging.info(size,  y_step, x_border_augment, y_border_augment, fw, th, k)
    logging.info('xgboost')
    logging.info('size %s',size)
    logging.info('x_step %s',x_step)
    logging.info('y_step %s',y_step)
    logging.info('x_border_augment %s',x_border_augment)
    logging.info('y_border_augment %s',y_border_augment)
    logging.info('th %s',th)
    logging.info('k %s',k)
    logging.info('Loading data ...')
    df_train = pd.read_csv('train.csv',
                           usecols=['row_id','x','y','time','place_id','accuracy'],
                           index_col = 0)
    df_test = pd.read_csv('test.csv',
                          usecols=['row_id','x','y','time','accuracy'],
                          index_col = 0)
    # df_validation = pd.read_csv('partTrain.csv',
    #                       usecols=['row_id', 'x', 'y', 'time', 'accuracy'],
    #                       index_col=0)
    #Feature engineering

    minute = df_train.time%60
    df_train['hour'] = df_train['time']//60
    df_train.drop(['time'], axis=1, inplace=True)
    df_train['weekday'] = df_train['hour']//24
    df_train['month'] = df_train['weekday']//30
    df_train['year'] = (df_train['weekday']//365+1)*10.0
    df_train['hour'] = ((df_train['hour']%24+1)+minute/60.0)*4.0
    df_train['weekday'] = (df_train['weekday']%7+1)*3.0
    df_train['month'] = (df_train['month']%12+1)*2.0
    df_train['accuracy'] = np.log10(df_train['accuracy'])*10.0

    minute = df_test['time']%60
    df_test['hour'] = df_test['time']//60
    df_test.drop(['time'], axis=1, inplace=True)
    df_test['weekday'] = df_test['hour']//24
    df_test['month'] = df_test['weekday']//30
    df_test['year'] = (df_test['weekday']//365+1)*10.0
    df_test['hour'] = ((df_test['hour']%24+1)+minute/60.0)*4.0
    df_test['weekday'] = (df_test['weekday']%7+1)*3.0
    df_test['month'] = (df_test['month']%12+1)*2.0
    df_test['accuracy'] = np.log10(df_test['accuracy'])*10.

    add_data = df_train[df_train.hour<6]
    add_data.hour = add_data.hour+96
    df_train = df_train.append(add_data)

    add_data = df_train[df_train.hour>98]
    add_data.hour = add_data.hour-96
    df_train = df_train.append(add_data)

    # print('Preparing validation data')
    # minute = df_validation['time'] % 60
    # df_validation['hour'] = df_validation['time'] // 60
    # df_validation['weekday'] = df_test['hour'] // 24
    # df_validation['month'] = df_test['weekday'] // 30
    # df_validation['year'] = (df_test['weekday'] // 365 + 1) * fw[5]
    # df_validation['hour'] = ((df_test['hour'] % 24 + 1) + minute / 60.0) * fw[2]
    # df_validation['weekday'] = (df_test['weekday'] % 7 + 1) * fw[3]
    # df_validation['month'] = (df_test['month'] % 12 + 1) * fw[4]
    # df_validation['accuracy'] = np.log10(df_test['accuracy']) * fw[6]
    # df_test.drop(['time'], axis=1, inplace=True)

    logging.info('Solving')
    #Solving classification problems inside each grid cell
    process_grid(df_train, df_test, size, x_step, y_step, x_border_augment, y_border_augment, fw, th, k)
    logging.info('finished**')
