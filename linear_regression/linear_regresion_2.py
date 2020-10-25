#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_size", default=10, type=int, help="Data size")
parser.add_argument("--range", default=10, type=int, help="Feature order range")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Create the data
    xs = np.linspace(0, 7, num=args.data_size)
    ys = np.sin(xs) + np.random.RandomState(args.seed).normal(0, 0.2, size=args.data_size)

    rmses = []
    for order in range(1, args.range + 1):

        data = np.transpose(np.array([np.power(xs,n) for n in range(1, order+1)]))


        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            data, ys,
            test_size=args.test_size, random_state=args.seed
        )

        ln = sklearn.linear_model.LinearRegression().fit(X_train, y_train)

        predict = ln.predict(X_test)

        mse = 1 / np.shape(y_test)[0] * np.sum((predict - y_test) * (predict - y_test))
        rmse = np.sqrt(mse)

        rmses.append(rmse)

    return rmses

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmses = main(args)
    for order, rmse in enumerate(rmses):
        print("Maximum feature order {}: {:.2f} RMSE".format(order + 1, rmse))