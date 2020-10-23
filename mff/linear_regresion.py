#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.9, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def lin_reg(x,y):

    first = np.linalg.inv(np.matmul(np.transpose(x) , x))
    second = np.matmul(np.transpose(x),y)
    return np.matmul(first,second)

def main(args):
    # Load Boston housing dataset
    dataset = sklearn.datasets.load_boston()

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        np.pad(dataset.data,((0,0),(0,1)),'constant', constant_values=(1)),dataset.target, test_size=args.test_size, random_state=args.seed
    )
    print(dataset.data)
    weight = lin_reg(X_train,y_train)


    predict = np.matmul(X_test,weight)

    mse =1/np.shape(y_test)[0] * np.sum((predict - y_test)*(predict - y_test))
    rmse = np.sqrt(mse)

    return rmse

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(args)
    print("{:.2f}".format(rmse))