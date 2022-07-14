import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import sympy


def plot_brain_landmarks(images, labels, preds, points, is_batch=True, z_first=True, plot_or_save_fig=True):

    if isinstance(images, torch.Tensor):
        images = images.numpy()

    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    if isinstance(preds, torch.Tensor):
        preds = preds.numpy()

    if isinstance(points, torch.Tensor):
        points = np.round(points.numpy()).astype(np.int)


    if not is_batch:
        images = [images]
        labels = [labels]
        preds = [preds]
        points = np.array([points])

    nbatches = points.shape[0]
    npoints = points.shape[1]
    dim = points.shape[2]

    labels_full = np.sum(labels, axis=1)
    preds_full = np.sum(preds, axis=1)


    for i in range(nbatches):
        if dim == 2:
            im_slice = images[i]
            points_2d = points
        else:
            z_fm = points[i, 2, 2]
            points_2d = points[i, :, :2]
            im_slice = images[i, 0] # 0 because images has 1 channel
            label_slice = labels_full[i]
            pred_slice = preds_full[i]
            if z_first:
                im_slice = images[i, 0, z_fm, :, :] # 0 because images has 1 channel
                label_slice = labels_full[i, z_fm, :, :]
                pred_slice = preds_full[i, z_fm, :, :]
            else:
                im_slice = images[i, 0, :, :, z_fm] # 0 because images has 1 channel
                label_slice = labels_full[i, :, :, z_fm]
                pred_slice = preds_full[i, :, :, z_fm]

        

        iml = sympy.Line(points_2d[0,:], points_2d[1,:])
        sp = sympy.Point2D(points_2d[2,:])
        project_sp_pred_2d = iml.projection(sp).evalf()


        # PLOT RESULTS
        plt.figure(figsize=[40,20])
        # plt.figure(figsize=[30,15])

        plt.subplot(131); plt.imshow(im_slice, cmap=cm.gray); plt.title('Image slice', fontsize=40)
        plt.tick_params(axis='both', which='major', labelsize=30)
        plt.axis('off')

        plt.subplot(132); plt.imshow(label_slice, cmap=cm.get_cmap('magma')); plt.title('Ground truth heat map', fontsize=40)
        plt.tick_params(axis='both', which='major', labelsize=30)
        plt.axis('off')

        plt.subplot(133); plt.imshow(im_slice, zorder=1, cmap=cm.gray)
        plt.imshow(pred_slice, zorder=2, alpha=0.6, cmap=cm.get_cmap('magma')); plt.title('Prediction', fontsize=40)
        plt.axis('off')

        plt.plot([points_2d[0,0], points_2d[1,0]], [points_2d[0,1], points_2d[1,1]], linewidth=4, zorder=3)
        plt.plot([points_2d[2,0], project_sp_pred_2d[0]], [points_2d[2,1], project_sp_pred_2d[1]], linewidth=4, zorder=3)

        plt.tick_params(axis='both', which='major', labelsize=30)
        # plt.tick_params(axis='both', which='minor', labelsize=20)

        if plot_or_save_fig:
            plt.show()
        else:
            plt.savefig('outputs/test_im_' + str(i) + '.jpg') # Run on cluster

        plt.close()


