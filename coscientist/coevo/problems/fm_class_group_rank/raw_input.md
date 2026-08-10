# Large ℓ-Rank in Class Groups of Imaginary Quadratic Fields (FrontierMath open problem)

- Domain: Algebraic number theory
- Type: Record Breaking
- Status: Unsolved (Epoch AI FrontierMath open-problems, `solved=False`)
- Source: https://epoch.ai/frontiermath/open-problems/class-group-rank

## About the problem

This problem asks for an algorithm to do the following: given an odd prime ℓ and a positive integer r, return a square-free positive integer D such that the number field K = Q(sqrt(-D)) has class group Cl_K with ℓ-rank >= r, i.e. dim_{F_ℓ} Cl_K / ℓ Cl_K >= r.

A paper of Cohen and Lenstra conjectures that for each ℓ, r there is a D such that Cl_K has ℓ-rank exactly r. In fact, they conjecture that there are infinitely many such D, and that such D form a positive proportion of all D. Nevertheless, there is no ℓ for which this conjecture is known.

For each ℓ there are infinitely many D that give rank >= 3 [Kim 2015], though already for rank 1 the known constructions yield D growing exponentially with ℓ, while the Cohen–Lenstra heuristics suggest D should grow only polynomially. For all but a few ℓ, no example of rank >= 4 is known.

For ℓ = 3, the current record rank is 8, due to Elkies in 2025. This was found not by direct search but via a connection with elliptic curves of high rank.

The hope is that a solution to this problem could contain ideas that help lead to a proof of the Cohen–Lenstra conjecture, i.e. that there is some D such that Cl_K has ℓ-rank r for each r.

The verifier tests a solution on a finite set of values of ℓ and r for which no D is known. Even a single such instance would likely be of interest.

## Prompt

Find imaginary quadratic fields K = Q(sqrt(-D)) whose class group Cl(K) has large ℓ-rank.

For a prime ℓ, the ℓ-rank of Cl(K) is the integer r such that the ℓ-torsion subgroup Cl(K)[ℓ] is
isomorphic to (Z/ℓZ)^r. Exhibit fields achieving ℓ-rank at least r for the (ℓ, r) pairs in the
challenge set below. You demonstrate that the ℓ-rank is at least r by giving r independent
class-group elements, each of order exactly ℓ, so that together they generate a subgroup
isomorphic to (Z/ℓZ)^r.

## Challenge set

Each (ℓ, r) pair below is a separate challenge:

    ℓ = 3    r = 9, 10, 11, 12, 13, 14, 15, 16
    ℓ = 5    r = 6, 7, 8, 9, 10, 11, 12
    ℓ = 7    r = 5, 6, 7, 8, 9, 10
    ℓ = 11   r = 4, 5, 6, 7, 8
    ℓ = 13   r = 4, 5, 6, 7, 8

Each challenge you certify earns credit; certifying every pair in the set earns full credit, and
any nonempty subset earns partial credit. Completing more is better.

## Submission

Write your answer to a file, one certificate per line, with fields separated by the pipe
character, and submit the file's path with the submit tool:

    ℓ|r|D|gen_1|gen_2|...|gen_r

- ℓ — the prime. Every generator on the line must have order exactly ℓ in Cl(K).
- r — the number of generators that follow. They must be independent.
- D — a positive integer such that -D is a fundamental discriminant; the field is K = Q(sqrt(-D)).
- gen_i — an element of Cl(K), given as a binary quadratic form of discriminant -D: three
  comma-separated integers "a,b,c" meaning a·x² + b·x·y + c·y², with b² - 4·a·c = -D.

Only lines whose (ℓ, r) is in the challenge set earn credit; you may include others (for example
the one below) but they are ignored.

### Example line

    3|3|4447704|390,-96,2857|921,-786,1375|346,-68,3217

This is ℓ = 3, r = 3, discriminant -4447704, and three independent order-3 forms — a well-formed
example of the format. (3, 3) is below the challenge set, so it verifies without counting.
