from sklearn.svm import LinearSVC

import pandas as pd
from evaluation import evaluate
import json
import numpy as np

def svm_out(w, x, b):
    return np.dot(w,x)+b

def classify_bin(w, b, data_test):
    return [svm_out(w, row, b) for row in data_test]

def classifiy(clf, b_list, data_test):
    result = []
    for row in data_test:
        selected_class = None
        max_svm = -float('inf')
        for key in clf:
            w = clf[key]
            b = b_list[key]
            res = svm_out(w, row, b)
            if (res>max_svm):
                max_svm = res
                selected_class = key
        result.append(selected_class)
    return result
def parseXY(dataframe: pd.DataFrame):
    coarse = dataframe['_coarse_'].values
    fine = dataframe['_fine_'].values
    val = dataframe.drop(columns=['_coarse_', '_fine_'])

    return val.values, coarse, fine ,val.columns

average = lambda x : sum(x)/len(x)

def find_evaluate(C, tol):
    file_name = ['input/fold-split-vect0.csv', 'input/fold-split-vect1.csv', 'input/fold-split-vect2.csv', 'input/fold-split-vect3.csv', 'input/fold-split-vect4.csv']

    df_list = [pd.read_csv(name) for name in file_name]
    result = []
    
    for i in range(5):
        print(f'Start fold {i} at C={C}, tol={tol}')
        test = df_list[i]
        train = pd.concat([df_list[j] for j in range(5) if i!=j])
        x_train, coarse_train, fine_train, label_train = parseXY(train)
        x_test, coarse_test, _, _ = parseXY(test)
        svm = LinearSVC(C=C, tol=tol, max_iter=2000, random_state=False, loss='hinge')
        svm.fit(x_train, coarse_train)
        print()
        clf = dict(zip(svm.classes_, svm.coef_.tolist()))
        b_list = dict(zip(svm.classes_, svm.intercept_.tolist()))
        classifier = dict((key,{'w':clf[key], 'b':-b_list[key]})for key in clf)
        svm_dict = {'C': C, 'eps':10**(-3), 'tol': tol, 'classifier': classifier}
        hasil = classifiy(clf, b_list, x_test)
        ev = evaluate(coarse_test, hasil, label=svm.classes_)
        
        # print(i, svm.get_param())
        hasil = svm.predict(x_test)
        ev = evaluate(coarse_test, hasil, label=sorted(set(hasil)))
        print(i, ev)
        
        f = open(f'output/fold-{i}-{C}-{tol}.json', 'w')
        json_str = json.dumps({'model': svm_dict, 'evaluation': ev}, indent=4)
        f.write(json_str)
        f.flush()
        f.close()
        result.append({
            'C': C,
            'tol': tol,
            'evaluation': ev
        })

    avg_eval = {
        'precision': 0,
        'recall': 0,
        'f_measure': 0,
        'accuracy': average([res['evaluation']['accuracy'] for res in result]),
        'avg_precision': average([res['evaluation']['avg_precision'] for res in result]),
        'avg_recall' : average([res['evaluation']['avg_recall'] for res in result]),
        'avg_f_measure': average([res['evaluation']['avg_f_measure'] for res in result])
    }

    temp = dict()
    for class_ in svm.classes_.tolist():
        temp[class_] = average([row['evaluation']['precision'][class_] for row in result])
    avg_eval['precision'] = temp
    temp = dict()
    for class_ in svm.classes_.tolist():
        temp[class_] = average([row['evaluation']['recall'][class_] for row in result])
    avg_eval['recall'] = temp
    temp = dict()
    for class_ in svm.classes_.tolist():
        temp[class_] = average([row['evaluation']['f_measure'][class_] for row in result])
    avg_eval['f_measure'] = temp

    avg_result = {
        'C': C,
        'tol': tol,
        'evaluation': avg_eval
    }
    f =open(f'output/avg-{C}-{tol}.json', 'w')
    json_str = json.dumps(avg_result, indent=4)
    f.write(json_str)
    f.flush()
    f.close() 
    return avg_result


def find_best_c():
    c_list = [0.5, 1, 1.5, 2, 2.5, 3]

    tol = 0.001
    all_param = []
    for c in c_list:
        temp = find_evaluate(c, tol)
        all_param.append(temp)
    
    all_param.sort(key = lambda x:(x['evaluation']['accuracy'], x['evaluation']['avg_f_measure']), reverse=True)
    best_param = all_param[0]    
    f =open(f'output/best_param_c.json', 'w')
    json_str = json.dumps(best_param, indent=4)
    f.write(json_str)
    f.flush()
    f.close() 

def find_best_tol():
    f = open('output/best_param_c.json', 'r')
    best_param = json.loads(f.read())
    f.close()
    c = best_param['C']
    tol_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    all_param = []
    for tol in tol_list:
        temp = find_evaluate(c, tol)
        all_param.append(temp)
    all_param.sort(key = lambda x:(x['evaluation']['accuracy'], x['evaluation']['avg_f_measure']), reverse=True)
    best_param = all_param[0]
    f =open(f'output/best_param_tol.json', 'w')
    json_str = json.dumps(best_param, indent=4)
    f.write(json_str)
    f.flush()
    f.close() 

if __name__=='__main__':
    find_best_c()
    find_best_tol()