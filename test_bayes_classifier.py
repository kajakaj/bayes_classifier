from sklearn import datasets
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from bayes_classifier import BayesClasifier


def test_with_k_fold_valid(data, target, k):
    kf = KFold(n_splits=k)
    count = 0 
    for train_index, test_index in kf.split(data):
        X_train = data[train_index]
        X_test = data[test_index]
        y_train = target[train_index]
        y_test = target[test_index]
        b = BayesClasifier(X_train, y_train)
        for i in range(len(X_test)):
            if b.test_classification(X_test[i], y_test[i]):
                count += 1
    return count/len(data)*100


def k_fold_valid_plot(data, target):
    x_s = []
    y_s = []
    for k in range(2, 20):
        x_s.append(k)
        y_s.append(test_with_k_fold_valid(data, target, k))
    plt.plot(x_s, y_s)
    plt.ylabel("accuracy [%]")
    plt.xlabel("k")
    plt.savefig('plot_k_fold.png')
    plt.clf()
    print(f'Best accuracy {max(y_s)} for k = {x_s[y_s.index(max(y_s))]}')


def main():
    iris = datasets.load_iris()
    
    # cross validation
    k_fold_valid_plot(iris.data, iris.target)
    

if __name__ == "__main__":
    main()
