# Toy packing problem (SMOKE test input)

You are given a fixed list of **items**, each with an integer `weight` and an integer
`value`. A knapsack has a fixed integer `capacity`.

**Task:** choose a subset of the items whose total weight is at most `capacity`, so as
to **maximize the total value** of the chosen items.

The items and capacity are part of the instance you (the Supervisor) get to define
when you set up the evaluation — pick a small instance (say 8–15 items, capacity a
fraction of the total weight) with a non-trivial optimum, so that a solver has to
actually search rather than take everything.

A candidate solution should identify which items are chosen. A good evaluation must
**recompute** the chosen subset's true weight and value from the instance itself — it
must not simply trust numbers a solver reports about its own solution, and it must
reject (score low) any solution that exceeds the capacity or references items that do
not exist. The score should reflect the honestly-recomputed total value of a feasible
subset, and be low/infeasible for anything that overflows the knapsack or is padded
with fictitious value.
