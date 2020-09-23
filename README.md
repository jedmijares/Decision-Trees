# Decision-Trees

This Python script implements the Iterative Dichotomiser 3 (ID3) algorithm to create decision trees.

## Synthetic Data

My first task was constructing decision trees based on the synthetic data located in the data directory. Each set is 2-dimensional data with a binary class identifier. I choose to store this data as a list of length 3 tuples, so that when I sorted the data based on one element, the other element and class label would stay attached appropriately. I then initially call my ID3 function with the list of data points, the index of the tuple containing the class label, and a list of available attributes indices to discriminate on - that is, all of them except the final class label element.

My ID3 function operates similarly to [the psuedocode available on Wikipedia](https://en.wikipedia.org/wiki/ID3_algorithm#Pseudocode). As for constructing the decision tree itself, nodes are given a `feature` value of the index of the attribute that best splits its data, calculated according to information gain. The children of that node are given a two item list, `values`, that represents the minimum and maximum value of that attribute that result in a data point belonging to this child's bin. This continues until one of the base cases of the ID3 algorithm has been met, at which point that node will be assigned a `label` of true or false.

### Plotting Data and Trees

For the synthetic data, I was asked to create plots mapping out our decision trees along with the given synthetic data. I adapted [scikit-learn's plotting example](https://scikit-learn.org/0.15/auto_examples/tree/plot_iris.html) to use the tree I had created, resulting in these plots:
