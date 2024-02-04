import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from code.src.bdt import compute_probability


def visualize_decision_boundary(x, y, predictor, ax=None, cmap=plt.cm.OrRd_r, ):
    nb = predictor()
    nb.fit(x=x, y=y)

    # Define a meshgrid to plot the decision boundary
    x_min, x_max = x.iloc[:, 0].min() - 1 / 3, x.iloc[:, 0].max() + 1 / 3
    y_min, y_max = x.iloc[:, 1].min() - 1 / 3, x.iloc[:, 1].max() + 1 / 3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Predict class labels for the points in the meshgrid
    Z = np.array(nb.predict(np.c_[xx.ravel(), yy.ravel()]))
    Z = np.array([list(np.unique(y)).index(_) for _ in Z])
    Z = Z.reshape(xx.shape)

    if ax is None:
        plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.6)
        unique_labels = np.unique(y)

        for label in unique_labels:
            indices = (y == label).values[:, 0]
            plt.scatter(x.iloc[indices, 0], x.iloc[indices, 1], label=f'Class {label}', edgecolor='k')

        plt.xlabel(x.iloc[:, 0].name)
        plt.ylabel(x.iloc[:, 1].name)
        plt.legend()
        plt.show()

    else:
        ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.6)
        unique_labels = np.unique(y)

        for label in unique_labels:
            indices = (y == label).values[:, 0]
            ax.scatter(x.iloc[indices, 0], x.iloc[indices, 1], label=f'Class {label}', edgecolor='k')

        ax.set_xlabel(x.iloc[:, 0].name)
        ax.set_ylabel(x.iloc[:, 1].name)
        ax.legend()


def visualize_prob_distribution(x, y):
    class_indices = list(y.groupby(list(y)).groups.values())
    number_line = np.linspace(-2, 9, 1000)
    stacked_num_line = np.vstack(
        [number_line, number_line, number_line, number_line]).transpose()  # a number line for each class
    random_sample = x.iloc[np.random.randint(0, len(x))]

    fig, axes = plt.subplots(2, 3, figsize=(22, 7), sharex=True)
    for _ in range(len(axes[0])):
        # Create and fit a prod_dist function
        probability_distribution = compute_probability(feature=x.iloc[class_indices[_]], thr=1e-3)
        approx = compute_probability(feature=x.iloc[class_indices[_]], thr=1e-3, integrate=False)

        # Plot

        axes[0][_].plot(number_line, approx(x=stacked_num_line))
        axes[0][_].scatter(random_sample, approx(x=np.array(random_sample)), label=str(random_sample))
        axes[0][_].set_title(f'Class: {y.iloc[class_indices[_]].iloc[0].values[0]}')
        axes[0][_].set_ylabel(f'Density')
        axes[0][_].legend()

        axes[1][_].plot(number_line, probability_distribution(x=stacked_num_line))
        axes[1][_].scatter(random_sample, probability_distribution(x=np.array(random_sample)), label=str(random_sample))
        axes[1][_].set_title(f'Class: {y.iloc[class_indices[_]].iloc[0].values[0]}')
        axes[1][_].set_ylabel(f'Probability')
        axes[1][_].legend()

    plt.show()


def visualize_confusion_matrix(pred, y, cmap=plt.cm.PuBu, ax=None, title=None):
    predictions = OrdinalEncoder().fit_transform(np.array((pred)).reshape(-1, 1))
    actual = OrdinalEncoder().fit_transform(np.array(y).reshape(-1, 1))
    cmat = confusion_matrix(actual, predictions)
    if ax is None:
        plt.imshow(cmat, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        for i in range(cmat.shape[0]):
            for j in range(cmat.shape[1]):
                plt.text(j, i, str(cmat[i, j]), ha='center', va='center', color='white')

        classes = [f'{np.unique(y)[i]}' for i in range(cmat.shape[0])]
        plt.xticks(np.arange(len(classes)), classes, rotation=45)
        plt.yticks(np.arange(len(classes)), classes)
        if title is not None:
            plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    else:
        img = ax.imshow(cmat, interpolation='nearest', cmap=cmap)
        plt.colorbar(img)
        for i in range(cmat.shape[0]):
            for j in range(cmat.shape[1]):
                ax.text(j, i, str(cmat[i, j]), ha='center', va='center', color='white')

        classes = [f'{np.unique(y)[i]}' for i in range(cmat.shape[0])]
        ax.set_xticks(np.arange(len(classes)), classes, rotation=45)
        ax.set_yticks(np.arange(len(classes)), classes)

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        if title is not None:
            ax.set_title(title)


def compute_metric_table(y_true, y_pred):
    encoder = OrdinalEncoder()

    recall = recall_score(y_true=encoder.fit_transform(np.array(y_true).reshape(-1, 1)),
                          y_pred=encoder.fit_transform(np.array(y_pred).reshape(-1, 1)), average=None)
    recall = pd.DataFrame(recall, index=np.unique(y_true))

    precision = precision_score(y_true=encoder.fit_transform(np.array(y_true).reshape(-1, 1)),
                                y_pred=encoder.fit_transform(np.array(y_pred).reshape(-1, 1)), average=None)
    precision = pd.DataFrame(precision, index=np.unique(y_true))

    f1 = f1_score(y_true=encoder.fit_transform(np.array(y_true).reshape(-1, 1)),
                  y_pred=encoder.fit_transform(np.array(y_pred).reshape(-1, 1)), average=None)
    f1 = pd.DataFrame(f1, index=np.unique(y_true))

    groups = list(y_true.groupby(list(y_true)).groups.values())
    true = OrdinalEncoder().fit_transform(np.array(y_true).reshape(-1, 1))
    p = OrdinalEncoder().fit_transform(np.array(y_pred).reshape(-1, 1))
    accuracy = pd.DataFrame([accuracy_score(true[_], p[_]) for _ in groups], index=np.unique(y_true))

    df = pd.merge(precision, recall, right_index=True, left_index=True)
    df = df.merge(f1, right_index=True, left_index=True)
    df.columns = ['precision', 'recall', 'f1']
    df = df.merge(accuracy, right_index=True, left_index=True)
    df.columns = ['precision', 'recall', 'f1', 'accuracy']

    return df