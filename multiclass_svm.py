from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from typing import Dict, Any
from bin_svm import (BinarySVM, KernelCache)
import numpy as np
import json
import fileout

class MultiClassSVM:
    @staticmethod
    def from_dict(param):
        C = param['C']
        tol = param['tol']
        max_iter = param['max_iter']

        svm = MultiClassSVM(C, tol, max_iter)
        classifier = dict()

        for key in param['classifier']:
            classifier[key] = BinarySVM.from_dict(param['classifier'][key])
        svm.classifier = classifier
        return svm

    def __init__(self, C, tol, max_iter=float('inf'), cache = None):
        self.C = C
        self.tol = tol
        self.classifier: Dict[Any, BinarySVM] = dict()
        self.kernel_cache: KernelCache = cache
        self.max_iter = max_iter
        
    def to_dict(self):
        clf = dict()

        for key in self.classifier:
            clf[f"{key}"] = self.classifier[key].to_dict()
        
        return {"C" : self.C, 'tol': self.tol, 'max_iter': self.max_iter, 'classifier': clf}

    def get_one_vs_all_classes(self, data_train, target):
        saved = dict()
        unique_target = set(target)
        for key in unique_target:
            key_target = [1 if x==key else -1 for x in target]
            saved[key] = key_target
        return saved

    def train(self, data_train, target):
        if (type(data_train) != np.ndarray):
            data_train = np.array(data_train)
        kernel_cache = KernelCache()
        separated = self.get_one_vs_all_classes(data_train, target)
        for class_ in separated:
            fileout.writeln("## TRAIN BINARY", class_)
            self.classifier[class_] = BinarySVM(self.C, self.tol, self.max_iter, kernel_cache)
            self.classifier[class_].train(data_train, separated[class_])
        self.kernel_cache = kernel_cache
        return self
    
    def classify(self, data):
        selected_class = None
        max_svm = -float('inf')
        for key in self.classifier:
            out = self.classifier[key].svm_out(data)
            if (out>max_svm):
                max_svm = out
                selected_class = key
        return selected_class

    def classify_all(self, data_test):
        res = []
        for row in data_test:
            res.append(self.classify(row))
        return res
    
    def score(self, data_test, target_test):
        classified = self.classify_all(data_test)
        accuracy = sum(1 for y1, y2 in zip(classified, target_test) if y1 == y2)/len(data_test)
        return accuracy
    def get_param(self):
        param = {}
        param['C'] = self.C
        param['eps'] = self.eps
        param['tol'] = self.tol
        param['classifier'] = dict((str(key), cl.get_param())for key, cl  in self.classifier.items())
        return param
    
if __name__== "__main__":
    x = load_iris().data
    y = load_iris().target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    clf = MultiClassSVM(1, 0.0001, 0.001)
    clf.train(x_train, y_train)
    print(clf.get_param())
    print(clf.score(x_test, y_test))
    out = open('out', 'w')
    out.write(json.dumps(clf.get_param(), indent=4))
    out.flush()
    clf = SVC(tol=0.001, kernel='linear').fit(x_train, y_train)
    print(clf.score(x_test, y_test))