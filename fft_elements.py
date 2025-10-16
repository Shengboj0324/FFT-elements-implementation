
import numpy as np
def fft_even_odd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT of even and odd indexed elements separately.
    Args:
        x: Complex vector input
    Returns:
        (FFT of even elements, FFT of odd elements)
    """
    even_elements = x[::2]
    odd_elements = x[1::2]
    fft_even = np.fft.fft(even_elements)
    fft_odd = np.fft.fft(odd_elements)
    return fft_even, fft_odd

def apply_twiddle_factors(x: np.ndarray, N: int, inverse: bool = False) -> np.ndarray:
    """
    Apply twiddle factors W_N^k = e^(-2πik/N) to complex vector.
    Args:
        x: Complex vector input
        N: FFT size (total number of points)
        inverse: If True, use positive exponent
    Returns:
        Vector with twiddle factors applied element-wise
    """
    k = len(x)
    indices = np.arange(k)
    sign = 1 if inverse else -1
    exponent = sign * 2j * np.pi * indices / N
    twiddle_factors = np.exp(exponent)
    return x * twiddle_factors

def combine_fft_even_odd(fft_even: np.ndarray, fft_odd: np.ndarray, N: int) -> np.ndarray:
    """
    Butterfly operation: combine even/odd FFT results using twiddle factors.
    Args:
        fft_even: FFT of even indexed elements
        fft_odd: FFT of odd indexed elements
        N: Total FFT size
    Returns:
        Combined FFT result: X[k] = E[k] + W_N^k*O[k], X[k+N/2] = E[k] - W_N^k*O[k]
    """
    twiddle_odd = apply_twiddle_factors(fft_odd, N, inverse=False)
    result = np.zeros(N, dtype=complex)
    half_N = len(fft_even)
    result[:half_N] = fft_even + twiddle_odd
    result[half_N:] = fft_even - twiddle_odd
    return result

def fft_radix2(x: np.ndarray) -> np.ndarray:
    """
    Recursive radix-2 FFT using Cooley-Tukey algorithm.
    Args:
        x: Complex vector (length must be power of 2)
    Returns:
        FFT of input vector
    Raises:
        ValueError: If input length is not power of 2
    """
    N = len(x)
    if N & (N - 1) != 0:
        raise ValueError(f"Input length must be power of 2, got {N}")
    if N == 1:
        return x

    # Split into even and odd elements
    even_elements = x[::2]
    odd_elements = x[1::2]

    # Recursively compute FFT
    fft_even = fft_radix2(even_elements)
    fft_odd = fft_radix2(odd_elements)

    # Combine using butterfly operation
    return combine_fft_even_odd(fft_even, fft_odd, N)

