# Decision Trees

This Python script implements the Iterative Dichotomiser 3 (ID3) algorithm to create decision trees.

## Synthetic Data

My first task was constructing decision trees based on the synthetic data located in the data directory. Each set is 2-dimensional data with a binary class identifier. I read this data with `readData()`, storing it as a list of length 3 tuples, so that when I sorted the data based on one element, the other element and class label would stay attached appropriately. I then initially call my ID3 function with the list of data points, the index of the tuple containing the class label, and a list of available attributes indices to discriminate on - that is, all of them except the final class label element.

My `ID3()` function operates similarly to [the psuedocode available on Wikipedia](https://en.wikipedia.org/wiki/ID3_algorithm#Pseudocode). As for constructing the decision tree itself, nodes are given a `feature` value of the index of the attribute that best splits its data, calculated according to information gain. The children of that node are given a two item list, `values`, that represents the minimum and maximum value of that attribute that result in a data point belonging to this child's bin. This continues until one of the base cases of the ID3 algorithm has been met, at which point that node will be assigned a `label` of true or false.

### Plotting Data and Trees

For the synthetic data, I was asked to create plots mapping out our decision trees along with the given synthetic data. I adapted [scikit-learn's plotting example](https://scikit-learn.org/0.15/auto_examples/tree/plot_iris.html) to use the tree I had created, resulting in these plots: 
![Plots](/media/plots.png)

### Bonus: Cross-Validation

As a bonus task, my script uses cross-validation to determine what maximum depth is ideal for each synthetic set's tree. I do this by splitting the data set into 4 folds. For each fold, I use the other 3 folds to create a tree and check the accuracy against the remaining, unused fold. I take the overall accuracy of each possible maximum depth and use the best performing maximum depth to create my final tree. This results in these plots: 
![Plots](/media/cross-validated-plots.png)

These plots seem identical to the previous plots, which is not surprising since there are only 2 features for each set, meaning the only real choice is whether to split once or twice. The first set is the only one to use a depth 1 tree, which makes sense as that is sufficient to perfectly classify the data.

## Pokémon Data

Included in the data folder is a .csv containing base stat, generation, and type data for over 1,400 Pokémon (including some made-up ones). Using this data, my script will create a decision tree of depth 3 to identify whether or not a Pokémon is "Legendary" given these stats. 

The same general process is used to construct this tree, though I made a few adjustments to support this. My `readData()` function takes an optional `hasHeader` argument to indicate if the 1st line of the .csv is the title of each feature, as the Pokémon includes headers while the synthetic sets do not. Additionally, the Pokémon set includes boolean data to indicate the typing of each Pokémon, as opposed to the purely numeric data of the synthetic sets. If a feature's header starts with "Type", then instead of setting each node's `values` attribute to the minimum and maximum value of its bin, I simply set `values` of one child to `[True, True]` and a second child's to `[False, False]`. Using a bin count of 8, this achieves 92.31% accuracy.

## Notes
Though it is typically good practice not to test on training data, for the purpose of this assignment, I do so.
While discretization is typically a pre-processing step, in this script I discretize the data into bins as I create the tree.
