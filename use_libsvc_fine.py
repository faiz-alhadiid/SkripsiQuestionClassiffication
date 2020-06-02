from sklearn.svm import SVC
from typing import Dict, List, Any
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from evaluation import evaluate
import json

def get_one_vs_all_classes(data_train, target):
    saved = dict()
    unique_target = set(target)
    for key in unique_target:
        key_target = [1 if x==key else 2 for x in target]
        saved[key] = key_target
    return saved
def svm_out(w, b, x):
    return np.dot(w, x) -b 

class MSVM:
    def __init__(self, C, tol, max_iter=-1):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.classifier: [Dict, SVC]= dict()
        self.weights = dict()
    

    def fit(self, x_train, y_train):
        separated = get_one_vs_all_classes(x_train, y_train)
        for class_ in separated:
            self.classifier[class_] = SVC(C=self.C, tol=self.tol, kernel='linear', max_iter=self.max_iter).fit(x_train, separated[class_])
            w = [x*-1 for x in self.classifier[class_].coef_[0]]
            b = self.classifier[class_].intercept_[0]
            b = b
            temp =  {
                'w': w,
                'b': float(b) 
            }
            self.weights[class_] = temp
    def predict_class(self, x_test_row):
        max_value = -float('inf')
        selected_class = None
        for key in self.weights:
            out = svm_out(self.weights[key]['w'], self.weights[key]['b'], x_test_row)
            if out> max_value:
                max_value = out
                selected_class = key
        return selected_class

    def predict(self, x_test):
        result = [self.predict_class(val) for val in x_test]
        return result
    def score(self, x_test, y_test):
        hasil = self.predict(x_test)
        n = len(hasil)
        return sum(1 for a, b in zip(y_test, hasil) if a==b)/n
    def get_param(self):
        param = {}
        param['C'] = self.C
        param['tol'] = self.tol
        param['max_iter'] = self.max_iter
        param['classifier'] = self.weights   
        return param  

def load_all_coarse_model():
    files = [
        'coarse-model-fold_0.json',
        'coarse-model-fold_1.json',
        'coarse-model-fold_2.json',
        'coarse-model-fold_3.json',
        'coarse-model-fold_4.json',
    ]

    result = []
    for file in files:
        name = './fine_test/coarse_model/'+file
        with open(name, 'r') as f:
            json_file = json.loads(f.read())
            model = json_file['model']
            clf = MSVM(model['C'], model['tol'], model['max_iter'])
            clf.weights = model['classifier']
            result.append(clf)
    return result

def split_by_coarse(x_train: np.ndarray, coarse_class:np.ndarray, fine_class: np.ndarray) -> Dict[Any, List]:
    saved = dict()
    unique = np.unique(coarse_class)
    for class_ in unique:
        indices = [i for i, c in enumerate(coarse_class) if c==class_]
        print(indices)
        saved[class_] = [x_train[indices], np.take(fine_class, indices), indices]
    return saved
class FineClassifier():
    def __init__(self, C, tol, max_iter):
        self.C =C
        self.tol = tol
        self.max_iter = max_iter
        self.classifier = dict()
    
    def fit(self, x_train, coarse, fine):
        fine_data = split_by_coarse(x_train, coarse, fine)
        for key, [x_fine, y_fine, indices] in fine_data.items():
            self.classifier[key] = MSVM(self.C, self.tol, self.max_iter)
            self.classifier[key].fit(x_fine, y_fine)
    
    def predict(self, x_test, coarse_result):
        result = []
        for i, row in enumerate(x_test):
            clf = self.classifier[coarse_result[i]]
            result.append(clf.predict_class(row))
        return result
    def get_param(self):
        param = dict((x, self.classifier[x].get_param()) for x in self.classifier)
        return param



def parseXY(dataframe: pd.DataFrame):
    coarse = dataframe['_coarse_'].values
    fine = dataframe['_fine_'].values
    val = dataframe.drop(columns=['_coarse_', '_fine_'])

    return val.values, coarse, fine ,val.columns

average = lambda x : sum(x)/len(x)

def save_klasifikasi_csv(name, actual, predicted):
    df =pd.DataFrame(data={'Aktual': actual, 'Predicted': predicted})
    df.to_csv('fine_test/classified/'+name+".csv", sep=';', index=False)

def find_evaluate(C, tol, max_iter):
    file_name = ['input/fold-split-vect0.csv', 'input/fold-split-vect1.csv', 'input/fold-split-vect2.csv', 'input/fold-split-vect3.csv', 'input/fold-split-vect4.csv']

    df_list = [pd.read_csv(name) for name in file_name]
    result = []
    coarse_classifier = load_all_coarse_model()
    for i in range(5):
        print(f'Start fold {i} at C={C}, tol={tol}, max_iter={max_iter}' )
        test = df_list[i]
        train = pd.concat([df_list[j] for j in range(5) if i!=j])
        x_train, coarse_train, fine_train, label_train = parseXY(train)
        x_test, coarse_test, fine_test, _ = parseXY(test)
        
        svm = FineClassifier(C, tol, max_iter)
        svm.fit(x_train, coarse_train, fine_train)
        
        coarse_result = coarse_classifier[i].predict(x_test)
        hasil = svm.predict(x_test, coarse_result)
        actual_class = [f"{c}:{f}"for c, f in zip(coarse_test, fine_test)]
        predicted_class = [f"{c}:{f}"for c, f in zip(coarse_result, hasil)]
        ev = evaluate(actual_class, predicted_class, label=sorted(set(actual_class)))
        print(i, ev)
        
        save_klasifikasi_csv(f'C_{C}-tol_{tol}-max_iter_{max_iter}-fold_{i}', actual_class, predicted_class)
        f = open(f'fine_test/C_{C}-tol_{tol}-max_iter_{max_iter}-fold_{i}.json', 'w')
        json_str = json.dumps({'model': svm.get_param(), 'evaluation': ev}, indent=4)
        f.write(json_str)
        f.flush()
        f.close()
        result.append({
            'C': C,
            'tol': tol,
            'max_iter': max_iter,
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

    label = set(actual_class)

    temp = dict()
    for class_ in label:
        temp[class_] = average([row['evaluation']['precision'][class_] for row in result])
    avg_eval['precision'] = temp
    temp = dict()
    for class_ in label:
        temp[class_] = average([row['evaluation']['recall'][class_] for row in result])
    avg_eval['recall'] = temp
    temp = dict()
    for class_ in label:
        temp[class_] = average([row['evaluation']['f_measure'][class_] for row in result])
    avg_eval['f_measure'] = temp

    avg_result = {
        'C': C,
        'tol': tol,
        'max_iter': max_iter,
        'evaluation': avg_eval
    }
    f =open(f'fine_test/AVG_C_{C}-tol_{tol}-max_iter_{max_iter}.json', 'w')
    json_str = json.dumps(avg_result, indent=4)
    f.write(json_str)
    f.flush()
    f.close() 
    return avg_result


def find_best_c():
    c_list = [0.5, 1, 1.5, 2, 2.5, 3]

    tol = 0.001
    max_iter = 500
    all_param = []
    for c in c_list:
        temp = find_evaluate(c, tol, max_iter)
        all_param.append(temp)
    
    all_param.sort(
        key = lambda x:(
            x['evaluation']['accuracy'], 
            x['evaluation']['avg_f_measure'],
            x['evaluation']['avg_precision'],
            x['evaluation']['avg_recall']), 
        reverse=True)
    best_param = all_param[0]    
    f =open(f'fine_test/best_param_c.json', 'w')
    json_str = json.dumps(best_param, indent=4)
    f.write(json_str)
    f.flush()
    f.close() 

def find_best_max_iter():
    f = open('fine_test/best_param_c.json', 'r')
    
    best_param = json.loads(f.read())
    
    f.close()
    c = best_param['C']
    
    tol = 0.001
    max_iter_list = [250, 500, 750, 1000, 1500, 2000]
    
    all_param = []
    for max_iter in max_iter_list:
        temp = find_evaluate(c, tol, max_iter)
        all_param.append(temp)
    all_param.sort(
        key = lambda x:(
            x['evaluation']['accuracy'], 
            x['evaluation']['avg_f_measure'],
            x['evaluation']['avg_precision'],
            x['evaluation']['avg_recall']), 
        reverse=True)
    best_param = all_param[0]
    f =open(f'fine_test/best_param_max_iter.json', 'w')
    json_str = json.dumps(best_param, indent=4)
    f.write(json_str)
    f.flush()
    f.close() 


def find_best_tol():
    f = open('fine_test/best_param_max_iter.json', 'r')
    best_param = json.loads(f.read())
    f.close()
    c = best_param['C']
    max_iter = best_param['max_iter']
    tol_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    all_param = []
    for tol in tol_list:
        temp = find_evaluate(c, tol, max_iter)
        all_param.append(temp)
    all_param.sort(
        key = lambda x:(
            x['evaluation']['accuracy'], 
            x['evaluation']['avg_f_measure'],
            x['evaluation']['avg_precision'],
            x['evaluation']['avg_recall']), 
        reverse=True)
    best_param = all_param[0]
    f =open(f'fine_test/best_param_tol.json', 'w')
    json_str = json.dumps(best_param, indent=4)
    f.write(json_str)
    f.flush()
    f.close() 

if __name__=='__main__':
    find_best_c()
    find_best_max_iter()
    find_best_tol()
    

