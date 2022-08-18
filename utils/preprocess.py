from scipy.signal import butter, iirnotch, filtfilt

def notch_filter(data, w0, q, fs, axis):
    # Get the filter coefficients 
    b, a = iirnotch(w0, q, fs)
    y = filtfilt(b, a, data, axis=axis)
    return y