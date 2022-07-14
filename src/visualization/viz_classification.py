# %%
#########################################################################################
# IMPORT
#########################################################################################

from IPython import get_ipython

# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

# # python.dataScience.interactiveWindowMode

# # %%
# seed = 100

# import os
# import pathlib
# import random
# random.seed(seed)
# import shutil
# # from os import path, listdir

import numpy as np
# np.random.seed(seed)
import pandas as pd
# # pd.random_state = seed

import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt

# plt.ioff()
# get_ipython().run_line_magic('matplotlib', 'agg')
# get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('seaborn-dark')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import seaborn as sns


# # %%
from PIL import Image
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc, confusion_matrix, average_precision_score, precision_recall_curve, f1_score




# # %%
# import torch

# # from torch.utils import data as D
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter


# import torchvision
# # import torchvision.transforms as transforms
# from torchvision import datasets, transforms, models
# import torchvision.transforms.functional as Ft
# import torch.nn.functional as Fn


# from torch.optim import lr_scheduler

# # from torchsummary import summary









# %%
# region CLASSIFICATION VISUALIZATION ###########################################
##################################################################


# %%
def plot_loss_accuracy(train_loss, val_loss, train_acc, val_acc, colors, 
                       loss_legend_loc='upper center', acc_legend_loc='upper left', 
                       fig_size=(20, 10), sub_plot1=(1, 2, 1), sub_plot2=(1, 2, 2)):
    
    plt.rcParams["figure.figsize"] = fig_size
    fig = plt.figure()
    
    plt.subplot(sub_plot1[0], sub_plot1[1], sub_plot1[2])
    
    for i in range(len(train_loss)):
        x_train = range(len(train_loss[i]))
        x_val = range(len(val_loss[i]))
        
        min_train_loss = train_loss[i].min()
        
        min_val_loss = val_loss[i].min()
        
        plt.plot(x_train, train_loss[i], linestyle='-', color='tab:{}'.format(colors[i]), 
                 label="TRAIN LOSS ({0:.4})".format(min_train_loss))
        plt.plot(x_val, val_loss[i], linestyle='--' , color='tab:{}'.format(colors[i]), 
                 label="VALID LOSS ({0:.4})".format(min_val_loss))
        
    plt.xlabel('epoch no.')
    plt.ylabel('loss')
    plt.legend(loc=loss_legend_loc)
    plt.title('Training and Validation Loss')
        
    plt.subplot(sub_plot2[0], sub_plot2[1], sub_plot2[2])
    
    for i in range(len(train_acc)):
        x_train = range(len(train_acc[i]))
        x_val = range(len(val_acc[i]))
        
        max_train_acc = train_acc[i].max() 
        
        max_val_acc = val_acc[i].max() 
        
        plt.plot(x_train, train_acc[i], linestyle='-', color='tab:{}'.format(colors[i]), 
                 label="TRAIN ACC ({0:.4})".format(max_train_acc))
        plt.plot(x_val, val_acc[i], linestyle='--' , color='tab:{}'.format(colors[i]), 
                 label="VALID ACC ({0:.4})".format(max_val_acc))
        
    plt.xlabel('epoch no.')
    plt.ylabel('accuracy')
    plt.legend(loc=acc_legend_loc)
    plt.title('Training and Validation Accuracy')
    
    fig.savefig('sample_loss_acc_plot.png')
    plt.show()
    
    return


# %%
def cm_analysis(y_true, y_pred, ymap=None, figsize=(10,10), to_show=False, filename='cfmat.jpg'):
    #  OUTPUT: PILLOW NUMPY IMAGE
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        # labels = [ymap[yi] for yi in labels]
        labels = ymap.values()
    else:
        labels = np.unique(y_true)

    n = len(labels)
    figsize = (1.5*n, 1.5*n)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)

    if to_show:
        # plt.savefig(filename)
        plt.show()
    else:
        fig.canvas.draw()

    # Now we can save it to a numpy array.
    cm_image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    print(cm_image.shape)
    cm_image = cm_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    print(cm_image.shape)

    return cm_image


# %%
def log_confusion_matrix(epoch, logs):
    test_pred_raw = model.predict(test_images)
    test_pred = np.argmax(test_pred_raw, axis=1)

    cm = confusion_matrix(test_labels, test_pred)
    figure = plot_confusion_matrix(cm, class_names=class_names)
    cm_image = plot_to_image(figure)

    # cm_image = cm_analysis(y_label.cpu().numpy(), y_pred.cpu().numpy(), file_name, labels, ymap=ymap, figsize=(5,5))

    with file_writer.as_default():
        writer.add_image("Confusion Matrix", cm_image, step=epoch)
        # writer.add_image('four_fashion_mnist_images', img_grid)

    # Define the per-epoch callback.
    # cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)


# %%
def cm_analysis_2(cm, labels, figsize=(10,10), to_show=False, filename='cfmat.jpg'):
    #  OUTPUT: PILLOW IMAGE
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    # if ymap is not None:
    #     y_pred = [ymap[yi] for yi in y_pred]
    #     y_true = [ymap[yi] for yi in y_true]
    #     labels = [ymap[yi] for yi in labels]

    # cm = confusion_matrix(y_true, y_pred, labels=labels)

    n = len(labels)
    figsize = (1.5*n, 1.5*n)

    # get_ipython().run_line_magic('matplotlib', 'inline')
    # get_ipython().run_line_magic('matplotlib', 'agg')
    # plt.ioff()

    # plt.rcParams.update({'font.size': round(1*n)})
    plt.rcParams.update({'font.size': 13})
    plt.rc('axes', titlesize=14)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)    # fontsize of the tick labels

    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    # fig, ax = plt.figure(figsize=figsize)
    
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)

    fig.canvas.draw()
    # plt.savefig(filename)

    if to_show:
        plt.show()
    # else:
    #     fig.canvas.draw()

    

    # Now we can save it to a numpy array.
    cm_image = np.array(fig.canvas.renderer.buffer_rgba())
    # print(cm_image.shape)
    # cm_image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # print(cm_image.shape)
    # cm_image = cm_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # print(cm_image.shape)

    im = Image.fromarray(cm_image)
    im = im.convert("RGB")
    # im.save(filename) 
    im = np.array(im)
    # print(im.shape)

    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111, frameon=False)
    # ax2.imshow(cm_image)
    # plt.show()

    # im = cm_image.transpose(2,0,1).float()/255.0
    # im = cm_image.permute(2,0,1).float()/255.0


    return im


# %%
def log_confusion_matrix_2(writer, epoch):

    cm_image = cm_analysis_2(cm, labels, figsize=(10,10), to_show=False, filename='cfmat.jpg')

    writer.add_image("Confusion Matrix", cm_image, step=epoch)



# endregion CLASSIFICATION VISUALIZATION