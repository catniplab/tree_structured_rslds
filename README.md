# tree_structured_rslds
Tree-structured recurrent switching linear dynamical systems (TrSLDS) are an extension of recurrent switching linear dynamical systems (rSLDS) from [Linderman et al., 2017](http://proceedings.mlr.press/v54/linderman17a/linderman17a.pdf).

Similar to rSLDS, TrSLDS introduces a dependency between the continuous and discrete latent states which allows the probability distribtuion of the discrete states to depend on the continuous states; this depdency paritiions the space, where each partition has it's own linear dynamics. While rSLDS partitions the space using (sequential) stick-breaking, TrSLDS utilizes tree-structured stick-breaking to partition the space:

![Stick-breaking](/aux/stick_breaking_tree.png)

A priori, it is natural to expect that locally linear dynamics of nearby regions in the latent space are similar. Thus,
in the context of tree-structured stick breaking, we impose that partitions that share a common parent
should have similar dynamics. We explicitly model this by enforcing a hierarchical prior on the dynamics that respects the tree structure which allows for a multi-scale view of the system. 
      
The model is efficenitly learned through Gibbs sampling. Details of the algorithm can be found [here](https://openreview.net/forum?id=HkzRQhR9YX&noteId=HJeZ36huRX).

# Installation
This package is built upon the following two packages:
````
github.com/slinderman/pypolyagamma
github.com/pytorch/pytorch
````

# Usage
To get started, please see the tutorial.
