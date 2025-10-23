
import numpy as np
import pytest
from fft_elements import fft_even_odd, apply_twiddle_factors, combine_fft_even_odd, fft_radix2


def test_fft_even_odd():
    x = np.array([1, 2, 3, 4], dtype=complex)
    fft_even, fft_odd = fft_even_odd(x)
    np.testing.assert_array_almost_equal(fft_even, np.fft.fft(x[::2]))
    np.testing.assert_array_almost_equal(fft_odd, np.fft.fft(x[1::2]))


def test_apply_twiddle_factors():
    x = np.ones(4, dtype=complex)
    result = apply_twiddle_factors(x, N=4)
    expected = np.array([1, -1j, -1, 1j])
    np.testing.assert_array_almost_equal(result, expected)


def test_apply_twiddle_inverse():
    x = np.ones(4, dtype=complex)
    result = apply_twiddle_factors(x, N=4, inverse=True)
    expected = np.array([1, 1j, -1, -1j])
    np.testing.assert_array_almost_equal(result, expected)


def test_combine_fft_even_odd():
    x = np.array([1, 0, 1, 0], dtype=complex)
    fft_even = np.fft.fft(x[::2])
    fft_odd = np.fft.fft(x[1::2])
    result = combine_fft_even_odd(fft_even, fft_odd, N=4)
    np.testing.assert_array_almost_equal(result, np.fft.fft(x))


def test_fft_radix2_basic():
    for n in [1, 2, 4, 8]:
        x = np.array(range(1, n+1), dtype=complex)
        result = fft_radix2(x)
        expected = np.fft.fft(x)
        np.testing.assert_array_almost_equal(result, expected)


def test_fft_radix2_random():
    for n in [2, 4, 8, 16]:
        x = np.random.randn(n) + 1j * np.random.randn(n)
        result = fft_radix2(x)
        expected = np.fft.fft(x)
        np.testing.assert_array_almost_equal(result, expected, decimal=10)


def test_fft_radix2_error():
    with pytest.raises(ValueError, match="power of 2"):
        fft_radix2(np.array([1, 2, 3], dtype=complex))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

