#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--plot", default=True, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.15, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Load Boston housing dataset
    dataset = sklearn.datasets.load_boston()

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        dataset.data, dataset.target,
        test_size=args.test_size, random_state=args.seed
    )
    print(dataset.data)


    lambdas = np.geomspace(0.01, 100, num=500)
    # TODO: Using `sklearn.linear_model.Ridge`, fit the train set using
    # L2 regularization, employing above defined lambdas.
    # For every model, compute the root mean squared error
    # (do not forget `sklearn.metrics.mean_squared_error`) and return the
    # lambda producing lowest test error.
    rmses = []
    best_lambda = 0
    mse = 100000
    for index,i in enumerate(lambdas):
        #print(i)
        model = sklearn.linear_model.Ridge(alpha=i).fit(X_train,y_train)
        predict = model.predict(X_test)
        mse_new = sklearn.metrics.mean_squared_error(y_test,predict)
        if mse_new < mse:
            best_lambda = i
            mse = mse_new
        rmses.append(np.sqrt(mse_new))

    best_rmse = min(rmses)



    if args.plot:
        # This block is not required to pass in ReCodEx, however, it is useful
        # to learn to visualize the results.

        # If you collect the respective results for `lambdas` to an array called `rmse`,
        # the following lines will plot the result if you add `--plot` argument.
        import matplotlib.pyplot as plt
        plt.plot(lambdas, rmses)
        plt.xscale("log")
        plt.xlabel("L2 regularization strength")
        plt.ylabel("RMSE")
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return best_lambda, best_rmse


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    best_lambda, best_rmse = main(args)
    print("{:.2f} {:.2f}".format(best_lambda, best_rmse))