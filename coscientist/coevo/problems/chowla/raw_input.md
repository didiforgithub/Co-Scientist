# Chowla's Cosine Problem

This is a genuinely open problem in analytic number theory / harmonic analysis.
There is no known complete solution. Below is a self-contained description. Nothing
else is provided — research, model, and formalize it yourself.

## Statement

Let $n_1 < n_2 < \dots < n_k$ be distinct positive integers. Form the cosine
polynomial
$$ f(x) = \sum_{j=1}^{k} \cos(n_j x), \qquad x \in \mathbb{R}. $$

Because $f$ is real, $2\pi$-periodic, even, and has mean value $0$ over a period
(each $\cos(n_j x)$ integrates to $0$), while $f(0) = k > 0$, the minimum
$$ M(n_1,\dots,n_k) \;=\; \min_{x \in \mathbb{R}} f(x) $$
is strictly negative.

Define the extremal quantity over all admissible integer sets of size $k$:
$$ m(k) \;=\; \max_{\substack{n_1<\dots<n_k \\ n_j \in \mathbb{Z}_{>0}}}\; \Big( \min_{x} \sum_{j=1}^{k}\cos(n_j x) \Big). $$
In words: choosing the $k$ distinct positive integers as cleverly as possible, how
close to $0$ (how *least negative*) can we force the minimum of the cosine sum to
be? Equivalently one studies $-m(k) = \min_{\text{sets}} \big(-\min_x f(x)\big) \ge 0$,
the smallest possible "dip" below zero.

## The problem

**Chowla's question / conjecture:** understand the growth of $-m(k)$ as
$k \to \infty$. Chowla conjectured that the minimum is forced to be very negative —
that $-m(k) \to \infty$, and moreover asked *how fast*.

What is known (you should verify and extend from the literature): it is a theorem
that $-m(k) \to \infty$; there are lower bounds of the form $-m(k) \gg k^{c}$ for
some small exponent, and much better bounds are conjectured. Determining the true
order of growth of $-m(k)$ is **open**. Related quantities: the same question for
the sum $\sum \cos(n_j x)$ restricted to structured sets (e.g. sets with small
difference sets / Sidon sets), and the analogous problem for $\sum \sin(n_j x)$.

## What a "solution" or "advance" could look like

Anything that genuinely moves the problem, for example:

- A **construction**: an explicit family of $k$-element integer sets achieving a
  provably (or numerically) large — i.e. least negative — value of $\min_x f(x)$,
  giving an upper bound on $-m(k)$.
- A **lower bound argument**: showing every $k$-set has $\min_x f(x) \le -g(k)$ for
  some growing $g$, ideally with a clean, checkable proof sketch.
- A **sharp numerical study** for small $k$ that reveals the extremal sets and the
  shape of $-m(k)$, with the reasoning made rigorous.
- A well-argued **opinion** on the true growth rate, with the supporting heuristics
  and the evidence that distinguishes it from alternatives.

Be honest about what is proven vs. conjectured vs. numerically suggested. Partial,
correct advances are worth far more than sweeping unjustified claims.
