"""Simple test runner (no pytest required)."""

import numpy as np
from fft_elements import fft_even_odd, apply_twiddle_factors, fft_radix2


def test_fft_even_odd():
    x = np.array([1, 2, 3, 4], dtype=complex)
    fft_even, fft_odd = fft_even_odd(x)
    assert np.allclose(fft_even, np.fft.fft(x[::2]))
    assert np.allclose(fft_odd, np.fft.fft(x[1::2]))
    print("✓ fft_even_odd")


def test_twiddle_factors():
    x = np.ones(4, dtype=complex)
    result = apply_twiddle_factors(x, N=4)
    expected = np.array([1, -1j, -1, 1j])
    assert np.allclose(result, expected)
    print("✓ apply_twiddle_factors")


def test_fft_radix2():
    for n in [1, 2, 4, 8, 16]:
        x = np.random.randn(n) + 1j * np.random.randn(n)
        result = fft_radix2(x)
        expected = np.fft.fft(x)
        assert np.allclose(result, expected, atol=1e-10)
    print("✓ fft_radix2 (sizes: 1, 2, 4, 8, 16)")


def test_error_handling():
    try:
        fft_radix2(np.array([1, 2, 3], dtype=complex))
        assert False
    except ValueError:
        pass
    print("✓ error handling")


def main():
    print("Running Core Tests")
    print("=" * 40)
    tests = [test_fft_even_odd, test_twiddle_factors, test_fft_radix2, test_error_handling]
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            return False
    print("=" * 40)
    print("All tests passed! ✓")
    return True


if __name__ == "__main__":
    exit(0 if main() else 1)

