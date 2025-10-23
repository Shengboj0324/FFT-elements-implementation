## Implementation Logic

**Decomposition Strategy**
- The input vector is split into even-indexed and odd-indexed elements at each recursion level
- Each subset undergoes independent FFT computation, reducing the problem size by half
- This decomposition continues recursively until reaching single-element base cases

**Twiddle Factor Application**
- Complex exponential weights W_N^k = e^(-2πik/N) are applied to the odd-indexed FFT results
- These phase rotations account for the frequency-domain relationships between interleaved time samples
- The inverse flag allows conjugate twiddle factors for inverse FFT operations

**Butterfly Combination**
- The even and odd FFT results are recombined using the butterfly operation
- First half: X[k] = E[k] + W_N^k × O[k] for k = 0 to N/2-1
- Second half: X[k+N/2] = E[k] - W_N^k × O[k] exploiting symmetry
- This exploits the periodicity of twiddle factors to avoid redundant computation

**Computational Efficiency**
- The recursive structure achieves O(N log N) complexity instead of O(N²) for naive DFT
- Each recursion level processes N elements with O(N) operations
- Total depth is log₂(N) for power-of-2 input lengths

## Functions

- `fft_even_odd(x)` - Separates and computes FFT of even/odd indexed subsets
- `apply_twiddle_factors(x, N, inverse)` - Applies complex exponential phase rotations
- `combine_fft_even_odd(fft_even, fft_odd, N)` - Executes butterfly recombination
- `fft_radix2(x)` - Main recursive FFT driver requiring power-of-2 length