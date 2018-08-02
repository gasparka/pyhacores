import matplotlib.pyplot as plt
import numpy as np



def load_iq(file):
    f = np.fromfile(open(str(file)), dtype=np.complex64)
    return f

def snr(pure, noisy):
    sig_pow = np.mean(np.abs(pure))
    error = np.array(pure) - np.array(noisy)
    err_pow = np.mean(np.abs(error))

    snr_db = 20 * np.log10(sig_pow / err_pow)
    return snr_db

def show_freqz(taps):
    from scipy import signal
    w, h = signal.freqz(taps)
    fig, ax1 = plt.subplots(1, 1)
    plt.title('Digital filter frequency response')
    ax1.plot(w / np.pi, 20 * np.log10(abs(h)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequency')
    plt.grid()
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    ax2.plot(w / np.pi, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    ax2.axis('tight')
    plt.tight_layout()
    plt.show()

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

