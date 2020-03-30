# Lecture notes on Static Superiority graph transformations

## Summary

We operate on the data dependence graph, where nodes are instructions.

The idea of Graph Transformations (GT) is to add edges (dependencies) between nodes
in a way that we do not prevent ourselves from finding an optimal solution.
These added edges make the scheduling problem easier, as an edge `A ---> B` tells
us that we do not need to check scheduling `B` before `A`.

If these GTs take polynomial time, or at least sub-exponential time, this can be
a win as compared to directly performing Branch and Bound.

## Static Superiority

Static Superiority is a GT approach which considers only register pressure.
This approach may result in schedules of sub-optimal length, but optimal register pressure.

Given a hypothetical scheduling problem with nodes `A` and `B`:
`A` is superior to `B` means that there exists an optimal schedule where `A` appears before `B`.

<sub>
N.B. our condition for superiority is fine, but more strict definitions could cause a problem:
`A superior B` and `B superior C` does not necessarily imply `A superior C`,
but adding the edges in the graph gives this implication.
</sub>

Our conditions for superiority:
Given a solution to this scheduling problem where `B` appears before `A`,
can we swap `A` and `B` without shutting out an optimal schedule? I.E. will swapping `A` and `B`
make things better or the same, but not worse?

There are conditions for legality of this (hypothetical) swap and conditions for optimality.

Definitions to simplify the following:

 - `Pred(N)`: recursive (aka transitive) predicessors of `N`. These are the instructions which `N` depends on.
 - `Succ(N)`: recursive successors of `N`. These are the instructions which depend on `N`.
 - `X ⊆ Y`: Mathematical subset operator: `X` is a subset of `Y`.

We are consider a hypothetical schedule which solves the scheduling problem, and swapping `A` and `B` as follows:

```
...                 ...
B                   A
...       -->       ...
A                   B
...                 ...
```

### Conditions for legality

 - `Pred(A) ⊆ Pred(B)`
 - `Succ(B) ⊆ Succ(A)`

### Conditions for optimality

We consider what happens to register pressure when swapping `A` and `B`.

#### Use point of view

 - Live ranges which are lengthened: those which are closed by `B`, i.e. whose last use is `B`.
 - Shortened live ranges: those which are closed by `A`.
 - If `r ∈ Use(A) ∩ Use(B)`, then the swap will not change the live range of `r`.
 - Therefore, if `Use(B) ⊆ Use(A)`, we can perform the swap. However, we can relax this further:
   - If every register `r` used by `B` but not by `A` is used by an instruction `C` where
     `C ≠ A` and `C ≠ B` and `C` is a recursive successor of `B`.

Translated into math:
 - Let `Rs = Use(B) - Use(A)`.
 - If the following holds for every `r ∈ Rs`:
   - There exists `C ∈ Succ(B)` such that `r ∈ Use(C)`

This condition works because `C` must appear after `A` in the hypothetical schedule,
meaning that `r` will still live until after the contested region even if `A` is moved earlier.

In fact, we can relax the above condition a little more to `C ∈ Succ(A)`,
as that implies that `C` must be outside of the contested region.

#### Def point of view

If, for each register type `T`, the number of registers defined by `A` is less than or equal to the number of registers defined by `B`.

I.E. For each register type `T`, `|Def(A)| ≤ |Def(B)|`.

#### Covering registers

For the Use POV, some of those registers whose live ranges are lengthened can be covered by the following:

 - Registers in `LastUse(A)`
 - Registers in defined by `B` but not by `A` (`|Def(B)| - |Def(A)|`).

We can prove that these registers' live ranges are shortened by the swap.
It's okay if some registers' live ranges are lengthened if at least as many registers' live ranges are
shortened.
