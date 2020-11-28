import matplotlib.pylab as plt
import numpy as np

'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


def h_line(y,
           ax=None,
           linestyle=":",
           linewidth=4,
           color="black",
           alpha=None,
           label=None):
    if ax is None: ax = get_axes()[0][0]
    ax.axhline(y, linestyle=linestyle, linewidth=linewidth, color=color, label=label, alpha=alpha)
    return ax


def bar_plot(y_vals,
             ax=None,
             x_vals=None,
             width=1,
             y_err=None,
             color=None,
             alpha=None,
             label=None):
    if ax is None: ax = get_axes()[0][0]
    if x_vals is None:
        ax.bar(np.arange(len(y_vals)), y_vals, yerr=y_err, width=width, color=color, label=label, alpha=alpha,
               capsize=3)
    else:
        ax.bar(x_vals, y_vals, yerr=y_err, width=width, color=color, label=label, alpha=alpha, capsize=3)
    return ax


def grouped_bar_plot(arr_vals,
                     ax=None,
                     legend_labels=None,
                     y_errs=None):
    if ax is None: ax = get_axes()[0][0]
    if y_errs is None: y_errs = [None] * len(arr_vals)
    if legend_labels is None: legend_labels = [None] * len(arr_vals)

    n_bars = len(arr_vals)
    gap_size = 1 / (n_bars + 1)
    spacing_x = np.linspace(-0.5, 0.5, num=n_bars + 1, endpoint=False)
    for i, vals in enumerate(arr_vals):
        ax.bar(np.arange(len(vals)) + spacing_x[i], vals, width=gap_size, yerr=y_errs[i], label=legend_labels[i])
    return ax


def line_plot(x_vals, y_vals,
              ax=None,
              y_errs=None,
              linewidth=None,
              bounds=None,
              alpha_err=0.1,
              label=None,
              color=None,
              alpha=1,
              linestyle="-"):
    if ax is None: ax = get_axes()[0][0]
    ax.plot(x_vals, y_vals, linestyle=linestyle, linewidth=linewidth, label=label, alpha=alpha, color=color)
    if y_errs is not None: ax.fill_between(x_vals, y_vals - y_errs, y_vals + y_errs, alpha=alpha_err, color=color)
    if bounds is not None: ax.fill_between(x_vals, bounds[0], bounds[1], alpha=alpha_err, color=color)
    return ax


def heatmap_plot(data,
                 ax=None,
                 value_display=False):
    if ax is None: ax = get_axes()[0][0]
    im = ax.imshow(data)
    ax.figure.colorbar(im, ax=ax)

    if value_display:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, data[i, j], ha="center", va="center", color="w")
    return ax


def scatter_plot(x_vals, y_vals,
                 ax=None,
                 label=None,
                 color=None,
                 size=5,
                 alpha=1):
    if ax is None: ax = get_axes()[0][0]
    ax.scatter(x_vals, y_vals, label=label, alpha=alpha, color=color, s=size)
    return ax


'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


def get_axes(rows=1, cols=1, sharex=False, sharey=False, figsize_x=10, figsize_y=5):
    _, axes = plt.subplots(rows, cols, figsize=(figsize_x, figsize_y * rows), squeeze=False, sharex=sharex,
                           sharey=sharey)
    return axes


def show():
    plt.show()


def savefig(fname):
    plt.savefig(fname)


def set_standard(ax,

                 title=None,
                 grid=False,
                 legend=False,

                 x_label=None,
                 x_lim=None,
                 x_scale_log=False,
                 x_tick_labels=None,

                 y_label=None,
                 y_lim=None,
                 y_scale_log=False,
                 y_tick_labels=None

                 ):
    if title is not None: ax.set_title(title)
    if grid: ax.grid()
    if legend: ax.legend()
    if x_label is not None: ax.set_xlabel(x_label)
    if y_label is not None: ax.set_ylabel(y_label)
    if x_lim is not None: ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim is not None: ax.set_ylim(y_lim[0], y_lim[1])
    if x_scale_log: ax.set_xscale("symlog")
    if y_scale_log: ax.set_yscale("symlog")
    if x_tick_labels is not None:
        ax.set_xticks(np.arange(len(x_tick_labels)))
        ax.set_xticklabels(x_tick_labels)
    if y_tick_labels is not None:
        ax.set_yticks(np.arange(len(y_tick_labels)))
        ax.set_yticklabels(y_tick_labels)
