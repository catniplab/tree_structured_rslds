# tree_structured_rslds
Tree-structured recurrent switching linear dynamical systems (TrSLDS) are an extension of recurrent switching linear dynamical systems (rSLDS) from [Linderman et al. AISTATS 2017](http://proceedings.mlr.press/v54/linderman17a/linderman17a.pdf).

Similar to rSLDS, TrSLDS introduces a dependency between the continuous and discrete latent states which allows the probability distribtuion of the discrete states to depend on the continuous states; this depdency paritiions the space, where each partition has it's own linear dynamics. While rSLDS partitions the space using (sequential) stick-breaking, TrSLDS utilizes tree-structured stick-breaking to partition the space.

! [Stick-breaking] (https://raw.githubusercontent.com/josuenassar/tree_structured_rslds/master/aux/stick_breaking_tree.pdf)
