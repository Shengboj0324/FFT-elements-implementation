# FFT Elements Implementation

## Files

**fft_elements.py** - Core implementation
- `fft_even_odd(x)`: Splits vector into even/odd indices, computes FFT of each subset
- `apply_twiddle_factors(x, N, inverse)`: Multiplies vector by W_N^k = e^(-2πik/N)
- `combine_fft_even_odd(fft_even, fft_odd, N)`: Butterfly operation combining even/odd FFTs
- `fft_radix2(x)`: Recursive Cooley-Tukey radix-2 FFT (requires power-of-2 length)

**test_fft_elements.py** - Pytest test suite
- 7 tests validating all functions against numpy.fft

**run_tests.py** - Standalone test runner (no pytest dependency)
- 4 core tests covering all functionality

## Installation

```bash
pip install numpy
```

## Testing

```bash
python run_tests.py
```

Or with pytest:
```bash
pytest test_fft_elements.py -v
```

## Algorithm

Radix-2 FFT via divide-and-conquer:
1. Split input into even-indexed and odd-indexed elements
2. Recursively compute FFT of each half
3. Combine using butterfly: X[k] = E[k] + W_N^k × O[k], X[k+N/2] = E[k] - W_N^k × O[k]

Where E[k] and O[k] are FFTs of even/odd elements, W_N^k = e^(-2πik/N) is the twiddle factor.