import numpy as np

def threshold_output(output, threshold=0.5):
    """
    Thresholds the output of the model to convert probabilities into binary values.

    Args:
    output (np.ndarray): The model's output as a NumPy array (values between 0 and 1).
    threshold (float): The threshold value for converting probabilities into binary values.

    Returns:
    np.ndarray: Binary output where values >= threshold are 1, and values < threshold are 0.
    """
    # Apply thresholding to convert probabilities to binary (0 or 1)
    binary_output = np.where(output >= threshold, 1, 0)
    
    return binary_output