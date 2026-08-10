# The Earth–Moon Problem (FrontierMath open problem)

- Domain: Graph theory
- Type: Construction - Finite; Improved Bound
- Status: Unsolved (Epoch AI FrontierMath open-problems, `solved=False`)
- Source: https://epoch.ai/frontiermath/open-problems/earth-moon

## About the problem

A graph is planar if it can be drawn in the plane such that no edges cross. A vertex-coloring of a
graph assigns every vertex a color such that the ends of every edge have different colors. The
chromatic number of a graph G is the minimum number of colors needed for a vertex-coloring of G.

The four color theorem says that every planar graph has chromatic number at most four.

The earth-moon problem asks about the chromatic number of graphs which can be edge-partitioned into
two planar graphs. A graph G = (V, E) is biplanar if there is a partition of its edges into two
sets E_1 and E_2 such that the graphs (V, E_1) and (V, E_2) are both planar.

Ringel, who introduced the earth-moon problem, proved in 1959 that the chromatic number of biplanar
graphs is at most 12. In 1980, Sulanke found a biplanar graph with chromatic number 9. These are
still the best-known upper and lower bounds on the chromatic number of biplanar graphs.

This problem asks for an improvement to the lower bound: specifically, a construction of a biplanar
graph with chromatic number at least 10. The model submits the edge-partition of the graph and a
vertex-coloring with 10 colors. The verifier checks that both parts are planar and that the
chromatic number is greater than 9.

The verifier has two main risks. First, it may be computationally infeasible to verify that the
chromatic number is greater than 9 on a very large graph. Second, it may be the case that biplanar
graphs have chromatic number at most nine, in which case the desired construction does not exist.

## Prompt

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
