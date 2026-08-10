# Low-peak trig design (SMOKE test input)

Design a real function on the interval `[0, 1]` of the form

    f(x) = sum over j of  a_j * cos(2*pi*k_j*x)

where you choose:
  * how many terms to use (between 2 and 8),
  * an integer frequency `k_j` in [1, 64] for each term (frequencies must be distinct),
  * a real amplitude `a_j` for each term.

The amplitudes are **normalized**: only their relative sizes matter, so require
`sum of |a_j| = 1` (a solution that does not satisfy this should be renormalized or
rejected).

**Objective:** make the function as *flat* as possible — MINIMIZE the peak magnitude

    peak(f) = max over x in [0, 1] of |f(x)|.

A lower peak is a better solution. (Intuitively, spreading energy across well-chosen
frequencies can lower the worst-case magnitude below that of any single tone.)

Note for whoever designs the evaluation: `peak(f)` is a maximum over a **continuous**
interval, which cannot be read off the coefficients directly and is genuinely
delicate to certify — the true peak can sit *between* any finite set of sample points,
so a solver could otherwise appear to score well while the real peak is much higher.
A candidate solution should report its chosen frequencies and amplitudes; a good
evaluation must compute the peak honestly from those coefficients (not trust any
peak value the solver reports about itself) and must not be foolable by a function
whose real peak hides between sampled points.
