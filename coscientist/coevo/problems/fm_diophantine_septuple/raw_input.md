# A Rational Diophantine Septuple (FrontierMath open problem)

- Domain: Diophantine
- Type: Integer Solution
- Status: Unsolved (Epoch AI FrontierMath open-problems, `solved=False`)
- Source: https://epoch.ai/frontiermath/open-problems/diophantine-septuple

## About the problem

A rational Diophantine n-tuple is a set a_1, ..., a_n of n positive rational numbers such that
a_i·a_j + 1 is the perfect square of a rational number for every i, j with 1 <= i < j <= n.
Rational Diophantine n-tuples were introduced by Diophantus around two thousand years ago. This
problem asks for a rational Diophantine 7-tuple.

Rational Diophantine n-tuples are known to exist for n <= 6, and it is unknown whether they exist
for n >= 7. Caporaso, Harris, and Mazur proved that if the Bombieri-Lang conjecture is true, then
for some positive integer m, there is no rational Diophantine n-tuple for n >= m, so it's likely
that rational Diophantine n-tuples don't exist for all n.

The primary risk for this verifier is that there may not exist a rational Diophantine 7-tuple;
otherwise, the verifier is an exact check. Another risk is that a solution could be brute-forced
without giving either an infinite family of examples or insight. The best solution might involve
constructing an elliptic curve with interesting or insightful properties, as Diophantine n-tuples
are often closely related to elliptic curves.

## Prompt

A *rational Diophantine m-tuple* is a set of m distinct non-zero rational numbers

    {a_1, ..., a_m}

with the property that a_i · a_j + 1 is the square of a rational number for every pair of distinct
indices i != j. Only the off-diagonal products are constrained: a_i^2 + 1 need not be a square.

## Task

Find a rational Diophantine 7-tuple: that is, 7 distinct non-zero rational numbers a_1, ..., a_7
such that a_i · a_j + 1 is the square of a rational number for every pair of distinct indices
i != j.

## Submission

Submit your 7 rationals using the submit tool, as a list of strings — each entry an integer "p" or
a fraction "p/q" (no decimal points or exponents). For example: ["1", "3", "8", "120", "-5/2"].
Your submission is checked exactly with arbitrary-precision rational arithmetic, so approximate or
floating-point values will not be accepted.
