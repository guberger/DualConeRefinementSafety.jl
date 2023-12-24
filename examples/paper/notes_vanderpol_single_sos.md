We explain the problems we encountered for computing a barrier function using
the SOS-based approach of Prajna et al. (2004).

* Stephen Prajna and Ali Jadbabaie. Safety verification of hybrid systems using
barrier certificates. Hybrid Systems: Computation and Control, 2004.

The code is in `exa_vanderpol_single_sos.jl`.

With degree 4, we tried different values of `\lambda` (0, 1, 1.5, 2, 2.5, 3).
None was giving a valid barrier function.

With degree 6, we tried different values of `\lambda` (0, 1, 1.5, 2, 2.5, 3).
The solver returned a solution claimed optimal, but a closer look at the points
on the extremities (e.g., `x \approx [-9.96, 0.2]`) demonstrated that those were
not valid barrier functions. The same happends by asking the Lie derivative +
`\lambda` the barrier to be strictly negative (e.g., `\espilonderiv = -0.0001`).
On the other hand, Z3 never finished for the verification.

With degree 8, we tried `\lambda=2` and `\espilonderiv = -0.0001`. We could find
something that looks valid. However, the high degree prevented Z3 from
validating the function (timed out after 12 hours).

# Version

Julia Version 1.9.3
Commit bed2cd540a (2023-08-24 14:43 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Windows (x86_64-w64-mingw32)
  CPU: 8 × 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, tigerlake)
  Threads: 1 on 8 virtual cores
Environment:
  JULIA_EDITOR = code
  JULIA_NUM_THREADS = 

MOSEK 10.1