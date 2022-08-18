import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def tsne(x, y, electrode_names, save_path):
    freq = y[:,0]
    amp = y[:,1]
    electrode = y[:,2:]
    electrode = (electrode * np.arange(1,electrode.shape[1]+1)).sum(1) 

    x_embedded = TSNE(n_components=2, perplexity=30).fit_transform(x)

    # freqeuncy
    fig, ax = plt.subplots()
    scatter = ax.scatter(x_embedded[:,0], x_embedded[:,1], c=freq, cmap="Spectral")
    legend = ax.legend(*scatter.legend_elements(), loc='lower left', title="freq")
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, 'tsne_f.png'))
    plt.close()

    # amplitude
    fig, ax = plt.subplots()
    scatter = ax.scatter(x_embedded[:,0], x_embedded[:,1], c=amp, cmap="Spectral")
    legend = ax.legend(*scatter.legend_elements(), loc='lower left', title="amp")
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, 'tsne_a.png'))
    plt.close()
    
    # electrode
    fig, ax = plt.subplots()
    # import pdb
    # pdb.set_trace()
    cmap = plt.cm.get_cmap("tab20", electrode_names.shape[0])
    for idx, (e, l) in enumerate(zip(np.unique(electrode), electrode_names)):
        indices = electrode == e
        ax.scatter(x_embedded[indices,0], x_embedded[indices,1], label=l, c=cmap(idx))
    ax.legend(loc='lower left', title="electrode")
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, 'tsne_e.png'))
    plt.close()

    # fig, ax = plt.subplots()
    # scatter = ax.scatter(x_embedded[:,0], x_embedded[:,1], c=electrode, cmap="tab20")
    # kw = dict(prop="colors", func=np.exp(x, 10))
    # legend = ax.legend(*scatter.legend_elements(**kw), loc='lower left', title="electrode")
    # fig.tight_layout()
    # plt.savefig(os.path.join(save_path, 'tsne_e.png'))
    # plt.close()

    
    # fig, ax = plt.subplots()
    # scatter = ax.scatter(x_embedded[:,0], x_embedded[:,1], c=electrode, cmap="tab20")
    # legend = ax.legend(*scatter.legend_elements(), loc='lower left', title="electrode")
    # fig.tight_layout()
    # plt.savefig(os.path.join(save_path, 'tsne_e.png'))
    # plt.close()

    # x_embedded = TSNE(n_components=3, perplexity=30).fit_transform(x)
    
    # # freqeuncy
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(x_embedded[:,0], x_embedded[:,1], x_embedded[:,2], c=freq, cmap="tab10")
    # legend = ax.legend(*scatter.legend_elements(), loc='lower left', title="freq")
    # fig.tight_layout()
    # plt.savefig(os.path.join(save_path, 'tsne_f.png'))
    # plt.close()

    # # amplitude
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(x_embedded[:,0], x_embedded[:,1], x_embedded[:,2], c=amp, cmap="tab10")
    # legend = ax.legend(*scatter.legend_elements(), loc='lower left', title="amp")
    # fig.tight_layout()
    # plt.savefig(os.path.join(save_path, 'tsne_a.png'))
    # plt.close()
    
    # # electrode
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(x_embedded[:,0], x_embedded[:,1], x_embedded[:,2], c=electrode, cmap="tab10")
    # legend = ax.legend(*scatter.legend_elements(), loc='lower left', title="electrode")
    # fig.tight_layout()
    # plt.savefig(os.path.join(save_path, 'tsne_e.png'))
    # plt.close()

# https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts