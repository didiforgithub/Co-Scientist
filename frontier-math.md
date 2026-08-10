# FrontierMath：5 道未解决题目

来源：[Epoch AI — FrontierMath: Open Problems](https://epoch.ai/frontiermath/open-problems) 及其官方 `Open Problems Data` ZIP（2026-07-31 更新）。整理日期：2026-08-11。

选择规则：按网页当前展示顺序，跳过已解决题目后取前 5 道标记为 **Unsolved** 的题目。题目名和 Prompt 来自官方 CSV 数据集；Description 来自各题详情页的 **About the problem** 全文。原文未翻译，以避免改变数学含义或提交格式。

许可：Epoch AI 在数据包 README 中声明该数据可在署名条件下依 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 使用、分发与复制。

---

## 1. A Baillie–PSW Pseudoprime

- 状态：Unsolved（官方数据集 `solved=False`）
- 领域：Number theory
- 类型：Counterexample; Integer Solution
- 重要性：Moderately interesting
- 原题页：[https://epoch.ai/frontiermath/open-problems/baillie-psw](https://epoch.ai/frontiermath/open-problems/baillie-psw)

### Description / About the problem（原文）

Testing whether or not a given positive integer is prime is a major problem with numerous applications. There are several heuristic tests that say that a number is probably prime. Two such tests are the strong Fermat prime test and the strong Lucas prime test. There are composite numbers (i.e. nonprime numbers) which are known to pass the strong Fermat prime test (see [here](https://oeis.org/A001262)) and the strong Lucas prime test (see [here](https://oeis.org/A217255)).

The **[Baillie-PSW primality test](https://en.wikipedia.org/wiki/Baillie%E2%80%93PSW_primality_test)** combines these two tests to say that a number is probably prime if it passes both the strong Fermat prime test and the strong Lucas prime test.

Carl Pomerance (the “P” in Baillie-PSW) [sketched](https://math.dartmouth.edu/~carlp/dopo.pdf) an argument suggesting that many composite numbers should pass the Baillie-PSW primality test. But to date, we do not know of a single non-prime number which passes the Baillie-PSW test. This problem asks for exactly such an example.

The Baillie-PSW test is widely used; for example, Maple and SageMath use Baillie-PSW to test whether a number is prime. A single counterexample would already advance the current state of understanding; the ability to generate many would be even stronger and could lead to better heuristic primality testing.

A solution to this problem is a positive integer \(x\) and two positive integers \(y\) and \(z\), not equal to \(1\) or \(x\), which multiply to \(x\). The verifier checks that \(y \cdot z = x\) and that \(x\) passes the Baillie-PSW primality test.

### Prompt（完整原文）

The Baillie-PSW primality test combines two probable-prime tests and declares an odd
integer n > 1 to be prime when it passes both. No composite integer is known to pass the Baillie-PSW test, and it has been verified that none exists below 2^64; whether any composite passes the
Baillie-PSW test is a long-standing open problem. Such a composite is called a Baillie-PSW
pseudoprime.

The two component tests are defined as follows.

Strong Fermat pseudoprime to base 2.
  Write n = d * 2^s + 1 with d odd. We say n is a strong Fermat
  pseudoprime to base 2 if either
      2^d ≡ 1 (mod n),   or
      2^(d * 2^r) ≡ -1 (mod n)  for some integer r with 0 ≤ r < s.
  (This is the strong, or Miller-Rabin, probable-prime test to base 2. See OEIS A001262.)

Strong Lucas pseudoprime (Selfridge Method A).
  Set P = 1. Let D be the first element of the sequence 5, -7, 9, -11, 13, ... (absolute value
  increasing by 2, sign alternating) such that the Jacobi symbol (D / n) equals -1, and set
  Q = (1 - D) / 4. Define the Lucas sequences U_k and V_k by
      U_0 = 0, U_1 = 1,  U_k = P*U_{k-1} - Q*U_{k-2},
      V_0 = 2, V_1 = P,  V_k = P*V_{k-1} - Q*V_{k-2}.
  Write n + 1 = e * 2^t with e odd. We say n is a strong Lucas pseudoprime if either
      U_e ≡ 0 (mod n),   or
      V_(e * 2^r) ≡ 0 (mod n)  for some integer r with 0 ≤ r < t.
  (See OEIS A217255.)

An odd composite n passes the Baillie-PSW test if it is BOTH a strong Fermat pseudoprime to
base 2 AND a strong Lucas pseudoprime.

Your task: find an odd composite integer n that passes the Baillie-PSW test.

Solution format:
* Write your answer to a JSON file and submit its path.
* The JSON must be an object with these integer keys:
  - "factor1", "factor2": two integers, each greater than 1. The candidate is their product
    n = factor1 * factor2; these factors certify that n is composite (they need not be prime).
    There is no bound on the size of the factors; arbitrary-precision arithmetic is used.

The verifier forms n = factor1 * factor2 with both factors > 1 (so n is a certified odd
composite), and then runs both component tests on n. Your submission is accepted if and only if
n is a certified odd composite that passes both tests.

Example submission (illustrating the format only — this n does not pass the test):
{
  "factor1": 23,
  "factor2": 89
}


---

## 2. Chowla's Cosine Problem

- 状态：Unsolved（官方数据集 `solved=False`）
- 领域：Analysis
- 类型：Construction - Finite; Improved Bound
- 重要性：Moderately interesting
- 原题页：[https://epoch.ai/frontiermath/open-problems/chowla-cosine](https://epoch.ai/frontiermath/open-problems/chowla-cosine)

### Description / About the problem（原文）

Given a set \(A\) of positive integers, we can define a cosine polynomial with respect to \(A\) as follows:
\[f_A(x) := \sum_{a \in A} \cos(ax).\]

**Chowla’s cosine problem**, a famous problem in analysis, asks about how negative such a function has to be. Specifically, the question is to bound the value of
\[-\max_{|A| = n} \left(\min_x f_A(x)\right).\]

Much work has gone into understanding lower bounds, including [two](https://arxiv.org/abs/2509.05260) [results](https://arxiv.org/abs/2509.03490) from the past [year](https://www.quantamagazine.org/networks-hold-the-key-to-a-decades-old-problem-about-waves-20260128/).

This problem asks for an improvement to the upper bound. In fact, it asks for something a bit stronger: an algorithm that will generate, for every given positive constant \(c\), a set \(A\) of positive integers such that \(f_A(x)\) is bounded below by \(-c\sqrt{|A|}\). This would disprove Chowla’s conjecture, which stated that \(\sqrt{|A|}\) is the right order for this value.

Since the 1960s, there are known examples of sets \(A\) that satisfy this bound for \(c = 1\). But there is no known example for \(c \leq \frac{1}{20}\).

Since we ask for an algorithm that will return a set \(A\) for any given \(c > 0\), the hope is that any solution to this problem will contain a new idea for how to construct such sets, and thus be interesting. The verifier uses a numerical sampling check, so there is a small risk there. There is also a risk that Chowla’s conjecture is true, in which case the problem is not solvable.

### Prompt（完整原文）

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


---

## 3. Large $\ell$-Rank in Class Groups of Imaginary Quadratic Fields

- 状态：Unsolved（官方数据集 `solved=False`）
- 领域：Algebraic number theory
- 类型：Record Breaking
- 重要性：Moderately interesting
- 原题页：[https://epoch.ai/frontiermath/open-problems/class-group-rank](https://epoch.ai/frontiermath/open-problems/class-group-rank)

### Description / About the problem（原文）

This problem asks for an algorithm to do the following: given an odd prime \(\ell\) and a positive integer \(r\), return a square-free positive integer \(D\) such that the [number field](https://en.wikipedia.org/wiki/Algebraic_number_field) \(K=\mathbb{Q}(\sqrt{-D})\) has [class group](https://en.wikipedia.org/wiki/Ideal_class_group) \(\operatorname{Cl}_K\) with \(\ell\)-rank \(\geq r,\) i.e. \(\dim_{\mathbb{F}_\ell} \operatorname{Cl}_K/\ell \operatorname{Cl}_K\geq r.\)

A [paper](https://pub.math.leidenuniv.nl/~lenstrahw/PUBLICATIONS/1984e/art.pdf) of Cohen and Lenstra conjectures that for each \(\ell,r\) there is a \(D\) such that \(\operatorname{Cl}_K\) has \(\ell\)-rank exactly \(r\). In fact, they conjecture that there are infinitely many such \(D\), and that such \(D\) form a positive proportion of all \(D\). Nevertheless, is no \(\ell\) for which this conjecture is known.

For each \(\ell\) there are infinitely many \(D\) that give rank \(\geq 3\) [Kim 2015], though already for rank \(1\) the known constructions yield \(D\) growing exponentially with \(\ell\), while the Cohen–Lenstra heuristics suggest \(D\) should grow only polynomially.  For all but a few \(\ell\), no example of rank \(\geq 4\) is known.

For \(\ell=3\), the current record rank is \(8\), due to [Elkies](https://link.springer.com/article/10.1007/s40993-024-00601-x) in 2025.  This was found not by direct search but via a connection with elliptic curves of high rank.

The hope is that a solution to this problem could contain ideas that help lead to a proof of the Cohen–Lenstra conjecture, i.e. that there is some \(D\) such that \(\operatorname{Cl}_K\) has \(\ell\)-rank \(r\) for each \(r\).

The verifier tests a solution on a finite set of values of \(\ell\) and \(r\) for which no \(D\) is known. Even a single such instance would likely be of interest.

### Prompt（完整原文）

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


---

## 4. A Rational Diophantine Septuple

- 状态：Unsolved（官方数据集 `solved=False`）
- 领域：Diophantine
- 类型：Integer Solution
- 重要性：Moderately interesting
- 原题页：[https://epoch.ai/frontiermath/open-problems/diophantine-septuple](https://epoch.ai/frontiermath/open-problems/diophantine-septuple)

### Description / About the problem（原文）

A **rational Diophantine \(n\)-tuple** is a set \(a_1, \ldots, a_n\) of \(n\) positive rational numbers such that \(a_ia_j + 1\) is the perfect square of a rational number for every \(i, j\) with \(1 \leq i < j \leq n\). Rational Diophantine \(n\)-tuples were introduced by Diophantus around two thousand years ago. This problem asks for a rational Diophantine \(7\)-tuple.

Rational Diophantine \(n\)-tuples are known to exist for \(n \leq 6\), and it is unknown whether they exist for \(n \geq 7\). Caporaso, Harris, and Mazur [proved](https://pubs.ams.org/JAMS/1997-10-01/S0894-0347-97-00195-1/) that if the Bombieri-Lang conjecture is true, then for some positive integer \(m\), there is no rational Diophantine \(n\)-tuple for \(n \geq m\), so it’s likely that rational Diophantine \(n\)-tuples don’t exist for all \(n\). See also [here](https://www.ams.org/journals/notices/201607/rnoti-p772.pdf) for a survey about Diophantine \(n\)-tuples and [here](https://web.math.pmf.unizg.hr/~duje/ref.html) for a long bibliography on rational Diophantine \(n\)-tuples.

The primary risk for this verifier is that there may not exist a rational Diophantine \(7\)-tuple; otherwise, the verifier is an exact check. Another risk is that a solution could be brute-forced without giving either an infinite family of examples or insight. The best solution might involve constructing an elliptic curve with interesting or insightful properties, as Diophantine \(n\)-tuples are often closely related to elliptic curves.

### Prompt（完整原文）

A *rational Diophantine m-tuple* is a set of m distinct non-zero rational numbers

    {a_1, ..., a_m}

with the property that a_i * a_j + 1 is the square of a rational number for every pair of distinct indices i != j. Only the off-diagonal products are constrained: a_i^2 + 1 need not be a square.

## Task

Find a rational Diophantine 7-tuple: that is, 7 distinct non-zero rational numbers a_1, ..., a_7 such that a_i * a_j + 1 is the square of a rational number for every pair of distinct indices i != j.

## Submission

Submit your 7 rationals using the submit tool, as a list of strings — each entry an integer "p" or a fraction "p/q" (no decimal points or exponents). For example: ["1", "3", "8", "120", "-5/2"]. Your submission is checked exactly with arbitrary-precision rational arithmetic, so approximate or floating-point values will not be accepted.


---

## 5. The Earth–Moon Problem

- 状态：Unsolved（官方数据集 `solved=False`）
- 领域：Graph theory
- 类型：Construction - Finite; Improved Bound
- 重要性：Moderately interesting
- 原题页：[https://epoch.ai/frontiermath/open-problems/earth-moon](https://epoch.ai/frontiermath/open-problems/earth-moon)

### Description / About the problem（原文）

A graph is **planar** if it can be drawn in the plane (or on a piece of paper) such that no edges cross. A **vertex-coloring** of a graph assigns every vertex a color in such a way that the ends of every edge have different colors. The **chromatic number** of a graph \(G\) is the minimum number of colors needed for a vertex-coloring of \(G\).

The four color theorem, one of the most famous theorems in graph theory, says that every planar graph has chromatic number at most four.

The [earth-moon problem](https://en.wikipedia.org/wiki/Earth%E2%80%93moon_problem) asks about the chromatic number of graphs which can be edge-partitioned into two planar graphs. A graph \(G = (V, E)\) is **biplanar** if there is a partition of its edges into two sets \(E_1\) and \(E_2\) such that the graphs \((V, E_1)\) and \((V, E_2)\) are both planar.

Ringel, who introduced the earth-moon problem, proved in 1959 that the chromatic number of biplanar graphs is at most 12. In 1980, Sulanke found a biplanar graph with chromatic number 9. These are still the best-known upper and lower bounds on the chromatic number of biplanar graphs. See [here](https://www.researchgate.net/publication/328519850_To_the_Moon_and_Beyond_Favorite_Conjectures_and_Open_Problems_-_2) for a survey and some history on the earth-moon problem.

This problem asks for an improvement to the lower bound: specifically, a construction of a biplanar graph with chromatic number at least 10. The model submits the edge-partition of the graph and a vertex-coloring with 10 colors. The verifier checks that both parts are planar and that the chromatic number is greater than 9.

The verifier has two main risks. First, it may be computationally infeasible to verify that the chromatic number is greater than 9 on a very large graph. Second, it may be the case that biplanar graphs have chromatic number at most nine, in which case the desired construction does not exist.

### Prompt（完整原文）

A graph G = (V, E) is *biplanar* if its edge set E can be partitioned into two parts E_1 and
E_2 such that both (V, E_1) and (V, E_2) are planar. (Equivalently, G has thickness at most 2.)

Your task: construct a biplanar graph whose chromatic number is at least 10. Such a graph
would improve the best known lower bound for the Earth-Moon problem from 9 to 10. (Achieving
12, the known upper bound, would resolve the problem completely.)

You must exhibit the graph as a partition of its edges into two planar parts E_1 and E_2,
supply a proper coloring of the graph, and state the chromatic number k you claim for it.
Your submission is accepted if and only if:

  1. Both (V, E_1) and (V, E_2) are planar (so G is biplanar);
  2. k is between 10 and 12 (every biplanar graph has chromatic number at most 12);
  3. the coloring you supply is a proper coloring of G = (V, E_1 ∪ E_2) using at most k colors
     (so the chromatic number of G is at most k); and
  4. G cannot be properly colored with k - 1 colors (so its chromatic number is at least k).

Conditions 3 and 4 together prove that the chromatic number of G is exactly k.

Solution format:
* Write your answer to a JSON file and submit its path.
* The JSON must be an object with these keys:
  - "num_vertices": a positive integer N. The vertices are the integers 0, 1, ..., N-1.
  - "edges_part1": a list of edges [u, v] forming the first planar part E_1.
  - "edges_part2": a list of edges [u, v] forming the second planar part E_2.
  - "coloring": a list of N integers; entry i is the color assigned to vertex i.
  - "chromatic_number": the integer k you claim is the chromatic number of G.
* E_1 and E_2 must be disjoint (together they are the edge set of G). Endpoints are integers
  in [0, N), and edges are undirected.
* Colors may be any integers; only which colors are equal matters. Color G with exactly k
  colors and set "chromatic_number" to that same k. Your choice of k must be at least 10 and
  at most 12. A larger value of k is a stronger result.

Example submission (illustrating the format only — this graph is not biplanar with a high
chromatic number):
{
  "num_vertices": 3,
  "edges_part1": [[0, 1], [1, 2]],
  "edges_part2": [[0, 2]],
  "coloring": [0, 1, 2],
  "chromatic_number": 3
}

