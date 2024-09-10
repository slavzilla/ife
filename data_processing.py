import numpy as np

def generate_signal(t, freq_modulation_func):
    """Generate signal based on time array and frequency modulation function."""
    return np.sin(2 * np.pi * freq_modulation_func(t) * t)

def perform_stft(x, window_sizes, N, NI):
    """Calculate STFT for multiple window sizes."""
    stfts = []
    for Nw in window_sizes:
        STFT = np.zeros((N - Nw, NI), dtype=np.complex64)
        for k in range(Nw//2, N - Nw//2):
            x1 = x[k - Nw//2 : k + Nw//2]
            STFT[k - Nw//2, :] = np.fft.fftshift(np.fft.fft(x1, NI))
        stfts.append(STFT)
    return stfts

def normalize_and_map(true_if, max_value):
    """Normalize and map true instantaneous frequency to pixel indices."""
    min_freq = true_if.min()
    max_freq = true_if.max()
    return (true_if - min_freq) / (max_freq - min_freq) * max_value

def unify_stfts(stfts):
    """
    Unify a list of STFT matrices into a single 3D numpy array.

    Each STFT is zero-padded and centered in the array.

    Args:
    stfts (list of np.array): List of 2D numpy arrays where each array is an STFT with potentially different dimensions.

    Returns:
    np.array: A 3D numpy array where each slice along the third axis is a zero-padded, centered STFT.
    """
    # Determine the maximum dimensions across all STFTs
    max_freq_bins = max(stft.shape[0] for stft in stfts)
    max_time_frames = max(stft.shape[1] for stft in stfts)
    num_stfts = len(stfts)
    
    # Create a unified 3D numpy array
    unified = np.zeros((max_freq_bins, max_time_frames, num_stfts), dtype=np.complex64)

    # Center and zero-pad each STFT in the unified array
    for idx, stft in enumerate(stfts):
        pad_freq = (max_freq_bins - stft.shape[0]) // 2
        pad_time = (max_time_frames - stft.shape[1]) // 2
        freq_slice = slice(pad_freq, pad_freq + stft.shape[0])
        time_slice = slice(pad_time, pad_time + stft.shape[1])

        unified[freq_slice, time_slice, idx] = stft

    return unified

def map_if_to_2d_image(if_arr, N):
    """
    Map an instantaneous frequency array to a 2D representation where each frequency 
    value is plotted at its corresponding time point along the diagonal.

    Args:
    if_arr (np.array): An array containing instantaneous frequency values.
    N (int): The dimension of the square 2D output array. This typically matches the length of `if_arr`.

    Returns:
    np.array: A 2D numpy array of shape (N, N) with frequency values mapped onto a 2D grid.
    """
    if_2d = np.zeros((N, N))

    # Loop over each element in the instantaneous frequency array
    for t in range(N):
        # Calculate the row index for the current time 't' by casting the frequency value to int
        row_index = int(if_arr[t])
        if_2d[row_index, t] = if_arr[t]

    return if_2d

import numpy as np

def modulate_and_add_noise(x, t, modulation_center=None, modulation_scale=None, noise_scale=None):
    """
    Apply Gaussian-like amplitude modulation centered at a specified time point and add Gaussian noise to a signal.

    Args:
    x (np.array): Original signal array.
    t (np.array): Time array corresponding to the signal.
    modulation_center (int, optional): Index of the time array t to center the modulation. If None, a random index is chosen.
    modulation_scale (float, optional): Controls the spread of the modulation. If None, a random value is chosen.
    noise_scale (float, optional): Controls the amplitude of the added noise. If None, a random value is chosen.

    Returns:
    np.array: The modulated and noise-added signal.
    """
    N = len(t)
    
    # Select a random time index if not specified
    if modulation_center is None:
        modulation_center = np.random.randint(0, N)
    
    # Generate a random modulation scale if not specified
    if modulation_scale is None:
        modulation_scale = np.square(np.random.rand())
    
    # Generate a random noise scale if not specified
    if noise_scale is None:
        noise_scale = np.random.rand()

    # Create the modulation envelope
    modulation = np.exp(-np.square((t - t[modulation_center])) * np.square(modulation_scale))
    
    # Apply modulation to the signal
    modulated_signal = x * modulation
    
    # Generate noise
    noise = noise_scale * (np.random.randn(N) + np.random.randn(N)) / np.sqrt(2)
    
    # Add noise to the modulated signal
    result_signal = modulated_signal + noise

    return result_signal