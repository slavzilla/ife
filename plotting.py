import matplotlib.pyplot as plt
import numpy as np

def plot_stfts(stfts, window_sizes):
    """Plot each STFT in a subplot."""
    plots = len(window_sizes)
    rows = 2
    columns = (plots + rows - 1) // rows
    fig, axs = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))

    for idx, STFT in enumerate(stfts):
        row_index = idx // columns
        col_index = idx % columns
        cax = axs[row_index, col_index].pcolor(np.abs(STFT).T, shading='auto')
        axs[row_index, col_index].set_title(f'Nw={window_sizes[idx]}')
        fig.colorbar(cax, ax=axs[row_index, col_index])
    
    plt.savefig('stft_plots.pdf')
    plt.close()

def plot_instantaneous_frequency(true_if_2d):
    """Plot the 2D map of the instantaneous frequency."""
    plt.imshow(true_if_2d, cmap='hot', aspect='auto', origin='lower')
    plt.colorbar(label='Intensity')
    plt.title('Instantaneous Frequency Representation')
    plt.xlabel('Time (samples)')
    plt.ylabel('Frequency Index')
    plt.savefig('instantaneous_frequency.pdf')
    plt.close()

def plot(x):
    plt.plot(x)
    plt.savefig('test.pdf')
    plt.close()