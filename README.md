# Master Thesis's Code

In this repository is possible to find the Python code used in my master's thesis. The main Python libraries for the analysis developed in this thesis are:
* [music21](https://web.mit.edu/music21/doc/about/referenceCorpus.html) for the musical side;
* [dionysus2](https://mrzv.org/software/dionysus2/) for persistent homology computations. 

On the python module `my_function.py` there are the function implemented using the above libraries.

## Clustering [YCAC](https://ycac.yale.edu/}) scores

The Yale-Classical Archives Corpus ([YCAC](https://ycac.yale.edu/})) is a thorough collection of musical pieces dating from Baroque to Romantic period. 
The total distribution of notes can be found on the csv file `all_distributions.csv`. This huge set of score is used in the python notebook `Test_with_random_scores_YCAC.ipynb` where is possible to perform a hierarchical clustering choosing a random number of scores for each composer.
