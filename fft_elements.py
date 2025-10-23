
import numpy as np
def fft_even_odd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Extract even-indexed elements (0, 2, 4, ...)
    even_elements = x[::2]
    # Extract odd-indexed elements (1, 3, 5, ...)
    odd_elements = x[1::2]
    # Compute FFT of even subset
    fft_even = np.fft.fft(even_elements)
    # Compute FFT of odd subset
    fft_odd = np.fft.fft(odd_elements)
    return fft_even, fft_odd

def apply_twiddle_factors(x: np.ndarray, N: int, inverse: bool = False) -> np.ndarray:
    k = len(x)
    # Create index array [0, 1, 2, ..., k-1]
    indices = np.arange(k)
    # Determine sign: +1 for inverse FFT, -1 for forward FFT
    sign = 1 if inverse else -1
    # Compute exponent: sign * 2πik/N
    exponent = sign * 2j * np.pi * indices / N
    # Compute twiddle factors: e^(exponent)
    twiddle_factors = np.exp(exponent)
    # Apply element-wise multiplication
    return x * twiddle_factors

def combine_fft_even_odd(fft_even: np.ndarray, fft_odd: np.ndarray, N: int) -> np.ndarray:
    # Apply twiddle factors to odd FFT: W_N^k * O[k]
    twiddle_odd = apply_twiddle_factors(fft_odd, N, inverse=False)
    # Allocate result array
    result = np.zeros(N, dtype=complex)
    half_N = len(fft_even)
    # First half: X[k] = E[k] + W_N^k * O[k]
    result[:half_N] = fft_even + twiddle_odd
    # Second half: X[k+N/2] = E[k] - W_N^k * O[k]
    result[half_N:] = fft_even - twiddle_odd
    return result

def fft_radix2(x: np.ndarray) -> np.ndarray:
    N = len(x)
    # Check if N is power of 2 using bitwise AND
    if N & (N - 1) != 0:
        raise ValueError(f"Input length must be power of 2, got {N}")
    # Base case: single element
    if N == 1:
        return x

    # Split into even-indexed elements
    even_elements = x[::2]
    # Split into odd-indexed elements
    odd_elements = x[1::2]

    # Recursively compute FFT of even subset
    fft_even = fft_radix2(even_elements)
    # Recursively compute FFT of odd subset
    fft_odd = fft_radix2(odd_elements)

    # Combine results using butterfly operation
    return combine_fft_even_odd(fft_even, fft_odd, N)