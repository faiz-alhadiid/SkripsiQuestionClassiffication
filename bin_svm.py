
from typing import List, Union
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import fileout

class KernelCache:
    def __init__(self):
        self.same = dict()
        self.diff = dict()

    def get(self, i1, i2) -> Union[float, None]:
        try:
            if (i1 == i2):
                return self.same[i1]

            x1, x2 = min(i1, i2), max(i1, i2)
            return self.diff[x1][x2]
        except:
            return None

    def add(self, i1, i2, value):
        if (i1 == i2):
            self.same[i1] = value
        else:
            inner = self.diff.get(i1)
            if (inner == None):
                inner = {i2: value}
            else:
                inner[i2] = value
            self.diff[i1] = inner

    def copy(self, indices):
        mapper = dict([(idx, i) for i, idx in enumerate(indices)])
        same_copy = dict()
        diff_copy = dict()
        for key in self.same:
            if key in mapper:
                same_copy[mapper[key]] = self.same[key]
        for key in self.diff:
            if key in mapper:
                inner = self.diff[key]
                inner_copy = dict([(mapper[idx], value)
                                for idx, value in inner.items() if idx in mapper])
                
                diff_copy[mapper[key]] = inner_copy
        kc = KernelCache()
        kc.same = same_copy
        kc.diff = diff_copy
        return kc


class BinarySVM:
    def __init__(self, C: float, tol: float, max_iter=1000, cache=KernelCache()):
        self.C = C
        self.tol = tol
        self.data_train: np.ndarray = None
        self.alpha: List[float] = None
        self.W: np.ndarray = None
        self.target = None
        self.b: float = 0
        self.errors: List[float] = None
        self.kernel_cache = cache
        self.max_iter = max_iter
        self.trained = False
    
    def to_dict(self) -> dict:
        return {'w': list(self.W), 'b': self.b, 'iteration': self.iteration}
    
    @staticmethod
    def from_dict(dict_obj):
        binn = BinarySVM(None, None)
        binn.W = np.array(dict_obj["w"])
        binn.b = dict_obj["b"]
        return binn


    def linear_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return x1.dot(x2)

    def get_kernel_value(self, i1: int, i2: int) -> float:
        kernel_val = self.kernel_cache.get(i1, i2)
        if (kernel_val != None):
            return kernel_val
        kernel_val = self.linear_kernel(
            self.data_train[i1], self.data_train[i2])
        self.kernel_cache.add(i1, i2, kernel_val)
        return kernel_val

    def svm_out(self, data: Union[np.ndarray, List[float]]) -> float:
        return self.W.dot(data) - self.b

    def take_step(self, i1: int, i2: int) -> bool:
        # print(i1+1, i2+1)
        if i1 == i2:
            return False

        y1 = self.target[i1]
        y2 = self.target[i2]
        E1 = self.errors[i1]
        E2 = self.errors[i2]
        a1 = self.alpha[i1]
        a2 = self.alpha[i2]

        s = y1*y2
        L, H = 0, 0
        if (y1 != y2):
            L = max(0, a2-a1)
            H = min(self.C, self.C + a2 - a1)
        else:
            L = max(0, a2 + a1 - self.C)
            H = min(self.C, a1 + a2)
        # print("L,H:",L, H)
        if (L == H):
            return False

        k11 = self.get_kernel_value(i1, i1)
        k12 = self.get_kernel_value(i1, i2)
        k22 = self.get_kernel_value(i2, i2)
        eta = k11 + k22 - 2 * k12

        if (eta > 0):
            a2_new = a2 + y2 * (E1 - E2)/eta

            if a2_new <= L:
                a2_new = L
            elif a2_new >= H:
                a2_new = H
        else:
            return 0
        if (abs(a2_new - a2) < 0.001*(a2_new + a2 + 0.001)):
            return 0
        a1_new = a1 + s * (a2 - a2_new)

        b1 = E1 + y1 * (a1_new - a1) * k11 + y2 * (a2_new - a2) * k12 + self.b
        b2 = E2 + y1 * (a1_new - a1) * k11 + y2 * (a2_new - a2) * k12 + self.b

        if 0 < a1_new and a1_new < self.C:
            b_new = b1
        elif 0 < a2_new and a2_new < self.C:
            b_new = b2
        else:
            b_new = (b1+b2)/2

        self.alpha[i1] = a1_new
        self.alpha[i2] = a2_new
        self.b = b_new

        # Update Weight
        self.W = (self.W + y1 * (a1_new - a1) * self.data_train[i1] +
                  y2*(a2_new - a2) * self.data_train[i2])
        # Update errors
        for i in range(self.N):
            self.errors[i] = self.svm_out(self.data_train[i]) - self.target[i]
        
        return True

    def examine_example(self, i2: int) -> int:
        y2 = self.target[i2]
        a2 = self.alpha[i2]
        E2 = self.errors[i2]
        r2 = E2 * y2

        if (r2 < - self.tol and a2 < self.C) or (r2 > self.tol and a2 > 0):
            non_zero_c_num = self.alpha[(self.alpha != 0) & (
                self.alpha != self.C)].size
            if (non_zero_c_num > 0):

                if E2 > 0:
                    i1 = np.argmin(self.errors)
                else:
                    i1 = np.argmax(self.errors)
                if self.take_step(i1, i2):
                    fileout.writeln(i1, i2, self.W, self.b)
                    return 1

            non_zero_c_index = []
            zero_c_index = []
            for i in range(self.N):
                if self.alpha[i] == 0 or self.alpha[i] == self.C:
                    zero_c_index.append(i)
                else:
                    non_zero_c_index.append(i)

            np.random.shuffle(non_zero_c_index)
            np.random.shuffle(zero_c_index)

            for i1 in non_zero_c_index:
                if self.take_step(i1, i2):
                    fileout.writeln(i1, i2, self.W, self.b)
                    return 1
            for i1 in zero_c_index:
                if (self.take_step(i1, i2)):
                    fileout.writeln(i1, i2, self.W, self.b)
                    return 1
        return 0

    def train(self, data_train: Union[np.ndarray, List[float]], target: Union[np.ndarray, List[int]]):
        if (type(data_train) != np.ndarray):
            data_train = np.array(data_train)

        self.data_train: np.ndarray = data_train
        self.target = target

        N, m = self.data_train.shape
        self.N = N
        self.m = m
        self.alpha = np.zeros(N)
        self.W = np.zeros(m)
        self.b = 0
        self.errors = np.zeros(N)

        examine_all = True
        num_changed = 0

        # Initial errors
        for i in range(N):
            self.errors[i] = self.svm_out(data_train[i]) - self.target[i]

        # Start Training
        it = 0
        while (num_changed > 0 or examine_all) and it < self.max_iter:

            num_changed = 0
            if examine_all:
                for i in range(N):
                    # print('out', [self.svm_out(self.data_train[j]) for j in range(N)])
                    # print('errors', self.errors)
                    num_changed += self.examine_example(i)

            else:
                for i in range(N):
                    if self.alpha[i] == 0 or self.alpha[i] == self.C:
                        continue
                    num_changed += self.examine_example(i)

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            it += 1
        self.trained = True
        self.iteration = it
        return self

    def get_param(self):
        param = {}
        param['w'] = self.W.tolist()
        param['b'] = self.b
        param['iter'] = self.iteration
        return param


def test():
    pass


if __name__ == '__main__':
    df = pd.read_csv('data.csv', header=None)
    dtrain = df.values
    svm = BinarySVM(1, 0.001)
    svm.train(dtrain, [1, 1, -1, -1, -1, -1])
    print('out', np.round([svm.svm_out(svm.data_train[j])
                           for j in range(6)], 6))
    print('errors', np.round(svm.errors, 6))
    print(svm.get_param())
