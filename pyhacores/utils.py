import matplotlib.pyplot as plt
import numpy as np


def show_plot():
    plt.tight_layout()
    plt.grid()
    if plt.gca().get_legend_handles_labels() != ([], []):
        plt.legend()
    plt.show()


def imshow(im, rescale=True):
    if rescale:
        from skimage.exposure import exposure
        p2, p98 = np.percentile(im, (2, 98))
        im = exposure.rescale_intensity(im, in_range=(p2, p98))

    plt.imshow(im, interpolation='nearest', aspect='auto', origin='lower')
    show_plot()

