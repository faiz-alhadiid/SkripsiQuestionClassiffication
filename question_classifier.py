from __future__ import annotations

from feat_extraction import feature_extraction, feature_combination_train, feature_combination_test
from multiclass_svm import MultiClassSVM
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import fileout
import json


def split_by_coarse(x_train: np.ndarray, coarse_class:np.ndarray, fine_class: np.ndarray) -> Dict[Any, List]:
    saved = dict()
    unique = np.unique(coarse_class)
    for class_ in unique:
        indices = [i for i, c in enumerate(coarse_class) if c==class_]
        print(indices)
        saved[class_] = [x_train[indices], np.take(fine_class, indices), indices]
    return saved

class QuestionClassifier():
    @staticmethod
    def load(filename) -> QuestionClassifier:
        file = open(filename, 'r')
        text = file.read()
        file.close()

        obj = json.loads(text)
        qc = QuestionClassifier()
        qc.label = obj['label']
        qc.coarse_classifer = MultiClassSVM.from_dict(obj['coarse'])
        
        fine = dict()
        fine_obj = obj['fine']

        for key in fine_obj:
            fine[key] = MultiClassSVM.from_dict(fine_obj[key])
        qc.fine_classifier = fine

        return qc
    
    def __init__(self, 
        c_coarse=1, tol_coarse=0.0001, max_iter_coarse = -float("inf"), 
        c_fine=1, tol_fine=0.0001,  max_iter_fine = -float("inf")):
        
        self.c_coarse = c_coarse
        self.tol_coarse = tol_coarse
        self.max_iter_coarse = max_iter_coarse
        self.c_fine = c_fine
        self.tol_fine = tol_fine
        self.max_iter_fine = max_iter_fine
        self.coarse_classifer: MultiClassSVM = MultiClassSVM(c_coarse, tol_coarse, max_iter_coarse)
        self.fine_classifier: Dict[Any,MultiClassSVM] = dict()
    
    def save(self, filename):
        coarse = self.coarse_classifer.to_dict()
        fine = dict( [(key, val.to_dict()) for key, val in self.fine_classifier.items()])
        label = self.label

        obj = {'coarse': coarse, 'fine': fine, 'label': label}

        if filename != None:
            file = open(filename, 'w+')
            file.write(json.dumps(obj, indent=4))
            file.flush()
            file.close()
        return obj

    def train(self, data_train, coarse_class, fine_class, skip_preprocessing = False, label = None):
        if (not skip_preprocessing):
            temp = feature_extraction(data_train)
            data_train, label = feature_combination_train(temp)
        
        # pd.DataFrame(data = data_train, columns=label).to_csv("fe_manualisasi.csv", sep=';', index=False)
        self.label = label
        fileout.writeln("### TRAIN COARSE CLASS")
        self.coarse_classifer.train(data_train, coarse_class)
        
        cache = self.coarse_classifer.kernel_cache
        fine_data = split_by_coarse(data_train, coarse_class, fine_class)
        print(fine_data)
        for key, [x_fine, y_fine, indices] in fine_data.items():
            print(key, indices)
            print(x_fine)
            fine_cache = cache.copy(indices)
            fileout.writeln("### TRAIN FINE", key)
            self.fine_classifier[key] = MultiClassSVM(self.c_fine, self.tol_fine, self.max_iter_fine, cache=fine_cache)
            self.fine_classifier[key].train(x_fine, y_fine)
        fileout.flush()        
        return self

    def classify(self, data_test, skip_preprocessing = False):
        if (not skip_preprocessing):
            temp= feature_extraction(data_test)
            data_test = feature_combination_test(temp, self.label)
        # pd.DataFrame(data=data_test, columns=self.label).to_csv('fe_manualisasi_test.csv', index=False, sep=';')
        result = []
        for row in data_test:
            coarse_result = self.coarse_classifer.classify(row)
            fine_result = self.fine_classifier[coarse_result].classify(row)
            result.append([coarse_result, fine_result] )
        return result

def test_if_works():
    data_train = [
        "Bagaimana cara kerja mesin fotokopi?",
        "Mengapa burung unta tidak bisa terbang?",
        "Kapan Hitler berkuasa di Jerman?",
        "Berapa banyak pasang sayap yang lalat tsetse miliki?",
        "Di mana titik tertinggi di Jepang?",
        "Apa negara Afrika yang didirikan oleh para budak Amerika bebas pada tahun 1847?"]
    data_test = [
        "Bagaimana cara kerja kalkulator ilmiah?", 
        "Berapa banyak manusia di bumi?", 
        "Apa air terjun tertinggi di Amerika Serikat?"]

    qc = QuestionClassifier(max_iter_coarse=1, max_iter_fine=1)
    qc.train(data_train, ['DESC', 'DESC', 'NUM', 'NUM', 'LOC', 'LOC'], ['Manner', 'Reason', 'Date', 'Count', 'Mount', 'Country'])
    print(qc.classify(data_test))

def is_fe_right():
    data_train = ["Bagaimana cara kerja mesin fotokopi?","Mengapa burung unta tidak bisa terbang?","Kapan Hitler berkuasa di Jerman?","Berapa banyak pasang sayap yang lalat tsetse miliki?","Di mana titik tertinggi di Jepang?","Apa negara Afrika yang didirikan oleh para budak Amerika bebas pada tahun 1847?"]
    data_test = ["Bagaimana cara kerja kalkulator ilmiah?", "Berapa banyak orang Yahudi yang dieksekusi di kamp konsentrasi selama Perang Dunia II?", "Apa air terjun tertinggi di Amerika Serikat?"]
    fe = feature_extraction(data_train)
    dt, label = feature_combination_train(fe)
    df = pd.DataFrame(data=dt, columns = label)
    df.to_csv('dt.csv', index=False)
    fe = feature_extraction(data_test)
    dt = feature_combination_test(fe, label)
    df = pd.DataFrame(data=dt, columns = label)
    df.to_csv('dte.csv', index=False)
if __name__=='__main__':
    test_if_works()
    pass

