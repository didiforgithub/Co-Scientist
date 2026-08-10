# Chowla's Cosine Problem (FrontierMath open problem)

- Domain: Analysis
- Type: Construction - Finite; Improved Bound
- Status: Unsolved (Epoch AI FrontierMath open-problems, `solved=False`)
- Source: https://epoch.ai/frontiermath/open-problems/chowla-cosine

## About the problem

Given a set A of positive integers, we can define a cosine polynomial with respect to A as follows:
    f_A(x) := sum_{a in A} cos(a x).

Chowla's cosine problem, a famous problem in analysis, asks about how negative such a function has to be. Specifically, the question is to bound the value of
    -max_{|A| = n} ( min_x f_A(x) ).

Much work has gone into understanding lower bounds, including two results from the past year.

This problem asks for an improvement to the upper bound. In fact, it asks for something a bit stronger: an algorithm that will generate, for every given positive constant c, a set A of positive integers such that f_A(x) is bounded below by -c*sqrt(|A|). This would disprove Chowla's conjecture, which stated that sqrt(|A|) is the right order for this value.

Since the 1960s, there are known examples of sets A that satisfy this bound for c = 1. But there is no known example for c <= 1/20.

Since we ask for an algorithm that will return a set A for any given c > 0, the hope is that any solution to this problem will contain a new idea for how to construct such sets, and thus be interesting. The verifier uses a numerical sampling check, so there is a small risk there. There is also a risk that Chowla's conjecture is true, in which case the problem is not solvable.

## Prompt

For a finite set A of positive integers, define the cosine polynomial

    f_A(x) = sum_{a in A} cos(a x).

It is important that A is a set, not a multiset: it contains no element more than once. Because the average of f_A over a full period is 0 while f_A(0) = |A|, the polynomial f_A must dip below 0 somewhere; the question is how negative it is forced to be, on the scale of sqrt(|A|).

The goal is to construct sets A for which f_A stays as close to non-negative as possible. Concretely:

    For every constant c > 0 and every positive integer n, construct a set A of positive integers with |A| >= n such that

        f_A(x) >= -c * sqrt(|A|)   for all real x.

## Your task

Write a Python script defining a function

    solution(c: float, n: int) -> str

that returns such a set A for the given c and n.

## Solution format

* The returned string must list the members of A as plain decimal positive integers separated by whitespace and/or commas, for example: "1 3 7 12". Members must be distinct.
* An example showing the required output format is provided in the file "solution_example.py".
