import numpy as np
from collections import Counter


class BayesClasifier:
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.classes = np.unique(self.target)
        self.n_classes_dict = Counter(self.target)
        self.n_col = len(self.data[0])
        self.col_class_val = self.get_col_class_val()


    def get_col_class_val(self):

        # creating dictionary {column : {class : values}}
        col_class_val = {}
        for row_num in range(len(self.data)):
            row_class = self.target[row_num]
            for col_num in range(self.n_col):
                if col_num in col_class_val.keys():
                    col_class_val[col_num][row_class].append(self.data[row_num][col_num])
                else:
                    new_col_dict = {}
                    for c in self.classes:
                        new_col_dict[c] = []
                    new_col_dict[row_class].append(self.data[row_num][col_num])
                    col_class_val[col_num] = new_col_dict

        return col_class_val


    def classifie(self, data):
        prob_dict = {}
        
        for c in self.classes:
            prob = self.n_classes_dict[c]/len(data)
            for col_num in range(self.n_col):
                var = np.var(self.col_class_val[col_num][c])
                mean = np.mean(self.col_class_val[col_num][c])
                prob *= 1/np.sqrt(2*np.pi*var) * np.exp(-(data[col_num]-mean)**2/(2*var))
            prob_dict[c] = prob
        
        return max(prob_dict, key=prob_dict.get)


    def test_classification(self, data, target):
        classification_result = self.classifie(data)
        return classification_result == target
