import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn import ensemble, cross_validation
from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import train_test_split
X, y = make_hastie_10_2(n_samples=5000)
X_train, X_test, y_train, y_test = train_test_split(X, y)

est = GradientBoostingClassifier(n_estimators=200, max_depth=3)
est.fit(X_train, y_train)
pred = est.predict(X_test)

acc = est.score(X_test, y_test)
#print('ACC: %.4f' % acc)
est.predict_proba(X_test)[0]

def ground_truth(x):
    """Ground truth -- function to approximate"""
    return x * np.sin(x) + np.sin(2 * x)

def gen_data(n_samples=200):
    """generate training and testing data"""
    np.random.seed(13)
    x = np.random.uniform(0, 10, size=n_samples)
    x.sort()
    y = ground_truth(x) + 0.75 * np.random.normal(size=n_samples)
    train_mask = np.random.randint(0, 2, size=n_samples).astype(np.bool)
    x_train, y_train = x[train_mask, np.newaxis], y[train_mask]
    x_test, y_test = x[~train_mask, np.newaxis], y[~train_mask]
    return x_train, x_test, y_train, y_test

X_train, X_test, y_train, y_test = gen_data(200)

# plot ground truth
x_plot = np.linspace(0, 10, 500)

def plot_data(figsize=(8, 5)):
    fig = plt.figure(figsize=figsize)
    gt = plt.plot(x_plot, ground_truth(x_plot), alpha=0.4, label='ground truth')

    # plot training and testing data
    plt.scatter(X_train, y_train, s=10, alpha=0.4)
    plt.scatter(X_test, y_test, s=10, alpha=0.4, color='red')
    plt.xlim((0, 10))
    plt.ylabel('y')
    plt.xlabel('x')


from sklearn.tree import DecisionTreeRegressor

est = DecisionTreeRegressor(max_depth=1).fit(X_train, y_train)
plt.plot(x_plot, est.predict(x_plot[:, np.newaxis]),
     label='RT max_depth=1', color='g', alpha=0.9, linewidth=2)

est = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)
plt.plot(x_plot, est.predict(x_plot[:, np.newaxis]),
     label='RT max_depth=3', color='g', alpha=0.7, linewidth=1)

plt.legend(loc='upper left')


from itertools import islice

est = GradientBoostingRegressor(n_estimators=1000,  max_depth=1, learning_rate=1.0)
est.fit(X_train, y_train)

ax = plt.gca()
first = True
for pred in islice(est.staged_predict(x_plot[:,np.newaxis]), 0, 1000, 10):
    plt.plot(x_plot, pred, color='r', alpha=0.2)
    if first:
        ax.annotate('High bias - low variance',
                    xy=(x_plot[x_plot.shape[0] // 2],
                        pred[x_plot.shape[0] // 2]),
                    xycoords='data',
                    xytext=(3, 4), textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                             connectionstyle="arc"))
        first = False

pred = est.predict(x_plot[:, np.newaxis])
plt.plot(x_plot, pred, color='r', label='GBRT max_depth=1')

ax.annotate('Low bias - high variance',
            xy=(x_plot[x_plot.shape[0] // 2],
            pred[x_plot.shape[0] // 2]),
            xycoords='data', xytext=(6.25, -6),
            textcoords='data',
            arrowprops=dict(arrowstyle="->",
            connectionstyle="arc"))
plt.legend(loc='upper left')

def rmspe(zip_list,count):
    # w = ToWeight(y)
    # rmspe = np.sqrt(np.mean((y - yhat) ** 2))
    sum_value=0.0
    # count=len(zip_list)
    for real,predict in zip_list:
        v1=(real-predict)**2
        sum_value += v1
    v2=sum_value / count
    v3=np.sqrt(v2)
    return v3

n_estimators = len(est.estimators_)
label=''
train_color='#2c7bb6'
test_color='#d7191c'
alpha=1.0

def deviance_plot(est, X_test, y_test, ax=None, label='', train_color='#2c7bb6',
              test_color='#d7191c', alpha=1.0):
    test_dev = np.empty(n_estimators)

    for i, pred in enumerate(est.staged_predict(X_test)):
       test_dev[i] = est.loss_(y_test, pred)

    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = plt.gca()

        ax.plot(np.arange(n_estimators) + 1, test_dev, color=test_color, label='Test %s' % label,
                 linewidth=2, alpha=alpha)
        ax.plot(np.arange(n_estimators) + 1, est.train_score_, color=train_color,
                 label='Train %s' % label, linewidth=2, alpha=alpha)
        ax.set_ylabel('Error')
        ax.set_xlabel('n_estimators')
        ax.set_ylim((0, 2))

    return test_dev, ax

test_dev, ax = deviance_plot(est, X_test, y_test,label='', train_color='#2c7bb6',
              test_color='#d7191c', alpha=1.0)
ax.legend(loc='upper right')

ax.annotate('Lowest test error', xy=(test_dev.argmin() + 1, test_dev.min() + 0.02), xycoords='data',
        xytext=(150, 1.0), textcoords='data',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc"),)

ann = ax.annotate('', xy=(800, test_dev[799]),  xycoords='data',
              xytext=(800, est.train_score_[799]), textcoords='data',
              arrowprops=dict(arrowstyle="<->"))
ax.text(810, 0.25, 'train-test gap')



# regularization
def fmt_params(params):
    return ", ".join("{0}={1}".format(key, val) for key, val in params.items())

fig = plt.figure(figsize=(8, 5))
ax = plt.gca()
for params, (test_color, train_color) in [({}, ('#d7191c', '#2c7bb6')),
                                      ({'min_samples_leaf': 3},
                                       ('#fdae61', '#abd9e9'))]:
    est = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=1, learning_rate=1.0)
    est.set_params(**params)
    est.fit(X_train, y_train)

    test_dev, ax = deviance_plot(est, X_test, y_test, ax=ax, label=fmt_params(params),
                             train_color=train_color, test_color=test_color)

ax.annotate('Higher bias', xy=(900, est.train_score_[899]), xycoords='data',
        xytext=(600, 0.3), textcoords='data',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc"),
        )
ax.annotate('Lower variance', xy=(900, test_dev[899]), xycoords='data',
        xytext=(600, 0.4), textcoords='data',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc"),
        )
plt.legend(loc='upper right')

## shrinkage

fig = plt.figure(figsize=(8, 5))
ax = plt.gca()
for params, (test_color, train_color) in [({},     ('#d7191c', '#2c7bb6')),
                                                  ({'learning_rate': 0.1},
                                       ('#fdae61', '#abd9e9'))]:
    est = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=1, learning_rate=1.0)
    est.set_params(**params)
    est.fit(X_train, y_train)

    test_dev, ax = deviance_plot(est, X_test, y_test, ax=ax, label=fmt_params(params),
                             train_color=train_color, test_color=test_color)

ax.annotate('Requires more trees', xy=(200, est.train_score_[199]), xycoords='data',
        xytext=(300, 1.0), textcoords='data',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc"),
        )
ax.annotate('Lower test error', xy=(900, test_dev[899]), xycoords='data',
        xytext=(600, 0.5), textcoords='data',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc"),
        )
plt.legend(loc='upper right')
plt.title("shrinkage")

fig = plt.figure(figsize=(8, 5))
ax = plt.gca()
for params, (test_color, train_color) in [({}, ('#d7191c', '#2c7bb6')),
                                      ({'learning_rate': 0.1, 'subsample': 0.5},
                                       ('#fdae61', '#abd9e9'))]:
    est = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=1, learning_rate=1.0,
                                random_state=1)
    est.set_params(**params)
    est.fit(X_train, y_train)
    test_dev, ax = deviance_plot(est, X_test, y_test, ax=ax, label=fmt_params(params),
                             train_color=train_color, test_color=test_color)

ax.annotate('Even lower test error', xy=(400, test_dev[399]), xycoords='data',
        xytext=(500, 0.5), textcoords='data',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc"),
        )

est = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=1, learning_rate=1.0,
                            subsample=0.5)
est.fit(X_train, y_train)
test_dev, ax = deviance_plot(est, X_test, y_test, ax=ax, label=fmt_params({'subsample': 0.5}),
                         train_color='#abd9e9',     test_color='#fdae61', alpha=0.5)
ax.annotate('Subsample alone does poorly', xy=(300, test_dev[299]), xycoords='data',
        xytext=(250, 1.0), textcoords='data',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc"),
        )
plt.legend(loc='upper right', fontsize='small')
plt.title("stochastic")
plt.show()


## find best hyper-parameters
from sklearn.model_selection import GridSearchCV

param_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
          'max_depth': [4, 6],
          'min_samples_leaf': [3, 5, 9, 17],
          # 'max_features': [1.0, 0.3, 0.1] ## not possible in our example (only 1 fx)
          }

est = GradientBoostingRegressor(n_estimators=3000)
# this may take some minutes
gs_cv = GridSearchCV(est, param_grid, n_jobs=4).fit(X_train, y_train)
print(gs_cv.best_params_)