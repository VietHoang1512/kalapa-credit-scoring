import os
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
from ast import literal_eval
from collections import Counter
from sklearn.metrics import auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb

def featureEngineer(train_full, test_full, submiss):

    test_full["label"] = -1
    data_full = pd.concat([train_full,test_full],sort=False)

    data_full.replace('Tỉnh Hòa Bình', 'Tỉnh Hoà Bình', inplace=True)
    data_full.replace('Tỉnh Vĩnh phúc', 'Tỉnh Vĩnh Phúc', inplace=True)

    data_full.maCv = data_full.maCv.fillna('missing')
    data_full.maCv = data_full.maCv.map(lambda string: string.lower(), na_action='ignore')

    #standardize macv

    data_full['job'] = None
    map_jobs = {('công nhân', 'cn', 'cnhân', 'lao động', 'may', 'c.nhân', 'coõng nhaõn', 'gò', 'máy', 'xưởng', 'khâu', 'dệt', 'lđ', 'ép', 'cao su', 'vận hành',) : 'công nhân',
                ('viên', 'nv', 'điện máy', 'trung cấp', 'nhõn viờn', 'nhan vien',) : 'nhân viên',
                ('vệ', 'gác', 'giám sát', 'trật tự', 'kiểm',) : 'bảo vệ',
                ('thiết kế',) : 'design',
                ('kỹ thuật', 'kt', 'trợ lý', 'thư ký', 'chuyên viên', 'trợ lí',) : 'vẫn là nhân viên nhưng kêu hơn',
                ('kỹ sư', 'phiên dịch', 'công chức', 'kĩ sư',) : 'trí óc',
                ('lái', 'tài xế',) : 'lái xe',
                ('điều dưỡng', 'y tế', 'y sỹ', 'dược', 'bác sỹ',) : 'y tế',
                ('chiến', 'bvệ',) : 'bộ đội',
                ('giáo viên', 'gv', 'gíao viên',) : 'giáo viên',
                ('bán hàng', 'sale', 'market', 'kinh doanh', 'bhàng', 'khách hàng', 'tư vấn',) : 'sale',
                ('kế toán', 'thủ quỹ', 'thủ quĩ',) : 'kế toán',
                ('tư pháp', 'hành chánh',) : 'hành chính hay đại khái thế',
                ('nợ', 'bar',) : 'bad guy',
                ('cử nhân', 'cử nhn',) : 'cử nhân',
                ('xây', 'xd', 'sơn', 'thợ', 'phụ', 'cơ khí sửa chữa', 'keo', 'đo đường',) : 'xây dựng',
                ('kế toán',) : 'kế toán',
                ('chủ tịch', 'bí thư', 'trưởng', 'đảng', 'phó', 'cán', 'quản lý', 'đốc', 'cb', 'ủy', 'xã đội',) : 'cán bộ'
                }
    for index, customer in data_full.iterrows():
        for macv_elems, job in map_jobs.items():
            if any(elem in customer.maCv for elem in macv_elems) :
                data_full.loc[index, 'job'] = job
    
    def string2list(string):
        return literal_eval(string)
    
    data_full.FIELD_7 = data_full.FIELD_7.fillna('[]')
    data_full.FIELD_7 = data_full.FIELD_7.map(string2list)

    #special column FIELD 7
    def special_count(df):
        elements = []
        for i in range(len(df)):
            elements += df[i]
        return list(set(elements))

    for element in special_count(data_full.FIELD_7):
        data_full['num_' + element] = 0*len(data_full)

    for index, customer in data_full.iterrows():
        count = Counter(customer['FIELD_7'])
        for elem, value in count.items():
            data_full['num_' + elem][index] += value

    data_full = data_full.drop(columns=['FIELD_7'])

    data_full['FIELD_11'] = pd.to_numeric(data_full['FIELD_11'], errors='coerce')
    data_full['FIELD_45'] = pd.to_numeric(data_full['FIELD_45'], errors='coerce')

    # word2number:
    word_count_cols = ['FIELD_35', 'FIELD_41', 'FIELD_42', 'FIELD_43', 'FIELD_44']
    
    map_count = {'Zero':0, "One":1, 'Two':2, 'Three':3, 'Four':4, 'Zezo':0, 
                      'None':-.5, 'Unknown':-1, 
                      'I':1, 'II':2, 'III':3, 'IV':4, 'V':5,
                      'A':1, 'B':2, 'C':3, 'D':4, '5':5, '0':0
                      }

    for column in word_count_cols:
        data_full[column] = data_full[column].map(map_count, na_action='ignore')  

    return data_full

def runLR(model, train_data, labels, test_data, index, n_folds=5, submiss_dir = './submiss'):
    fig, ax = plt.subplots()
    aucs = []
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True)

    for i, (train,valid) in enumerate(cv.split(train_data, labels)):

        model.fit(train_data[train], labels[train])
        plot = plot_roc_curve(model, train_data[valid], labels[valid], name=f'Fold number {i+1}', ax=ax)
        aucs.append(plot.roc_auc)
        test_pred = model.predict_proba(test_data)[:,1]
        
        submiss = pd.DataFrame({"id":index, "label": test_pred})
        submiss_path = os.path.join(submiss_dir, f'Logistic_Regression_{plot.roc_auc:.2f}_{i+1}.csv')
        submiss.to_csv(submiss_path, index=False)

    ax.plot([0,1], [0,1], label='Luck', linestyle='--', color='r')  
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    ax.plot(mean_auc, label=f'Average AUC score: {mean_auc:.2f} $\pm$ {std_auc:.2f}')  
    ax.legend(loc="lower right")
    ax.set(xlim=[-.1, 1.1], ylim=[-.1, 1.1], title='Logistic Regression')
    plt.show()	


def runXGB(model, train_data, labels, test_data, index, n_folds=5, submiss_dir = './submiss'):

    fig, ax = plt.subplots()
    aucs = []
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for i, (train,valid) in enumerate(cv.split(train_data, labels)):
        model.fit(train_data[train], labels[train], 
                 early_stopping_rounds=50, 
                 eval_set=[(train_data[valid], labels[valid])], 
                 verbose=0)
        plot = plot_roc_curve(model, train_data[valid], labels[valid], name=f'Fold number {i+1}', ax=ax)
        aucs.append(plot.roc_auc)
        test_pred = model.predict_proba(test_data)[:,1]
        
        submiss = pd.DataFrame({"id":index, "label": test_pred})
        submiss_path = os.path.join(submiss_dir, f'XGB_{plot.roc_auc:.2f}_{i+1}.csv')
        submiss.to_csv(submiss_path, index=False)

    ax.plot([0,1], [0,1], label='Luck', linestyle='--', color='r')  
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    ax.plot(mean_auc, label=f'Average AUC score: {mean_auc:.2f} $\pm$ {std_auc:.2f}')  
    ax.legend(loc="lower right")
    ax.set(xlim=[-.1, 1.1], ylim=[-.1, 1.1], title='XGBoost Classifier')
    plt.show()

def runLGB(model, train_data, labels, test_data, index, n_folds=5, submiss_dir = './submiss'):
    fig, ax = plt.subplots()
    aucs = []
    cv = StratifiedKFold(n_splits=5, shuffle=True)

    for i, (train,valid) in tqdm_notebook(enumerate(cv.split(train_data, labels))):

        dtrain = lgb.Dataset(train_data.loc[train], label=labels[train])
        dvalid = lgb.Dataset(train_data.loc[valid], label=labels[valid])
        model = lgb.train(params,  dtrain, valid_sets = [dvalid], verbose_eval=False, early_stopping_rounds=50, )

        preds = model.predict(train_data.loc[valid])
        fpr, tpr, threshold = roc_curve(labels[valid], preds)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, label = f'Fold number {i+1} (AUC = {roc_auc:.2f})')
        
        test_pred = model.predict(test_data)
        submiss = pd.DataFrame({"id":index, "label": test_pred})
        submiss_path = os.path.join(submiss_dir, f'LGB_{roc_auc:.2f}_{i+1}.csv')
        submiss.to_csv(submiss_path, index=False)

    ax.plot([0,1], [0,1], label='Luck', linestyle='--', color='r')  
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    ax.plot(mean_auc, label=f'Average AUC score: {mean_auc:.2f} $\pm$ {std_auc:.2f}')  
    ax.legend(loc="lower right")
    ax.set(xlim=[-.1, 1.1], ylim=[-.1, 1.1], title='Light GBM')
    plt.show()