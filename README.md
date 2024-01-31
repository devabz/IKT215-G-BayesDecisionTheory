# Bayes Decision Theory

In this assignment we'll construct a Naive Bayes Classifier, using several ideas, but focus on to very important ideas from bayesian probability theory.
The first being **independence**, which in layman terms, essentially refers to the property of a probability for 
one thing occurring is unrelated and unaffected of the probability of another thing occurring. The second being **Bayes Theorem**,
which essentially describes the relationship between single probabilities and conditional probabilities. 

# Independence
## Exclusivity
Independence depends on the fact that the outcomes or possibilities of whatever space, universe or phenomenon we're interested in, are mutually exclusive. 
Essentially, in such a universe, any sample maps to a single outcome among many possible outcomes.
<br><br>
Formally, we can describe exclusiveness using set theory. A series of events $ A_1, A_2, ..., A_n $ are mutually exclusive if, and only if


$$ A_i \cap A_j = A_iA_j = \emptyset \quad \text{for all} \; i \neq j $$

![img.png](images/img.png)
*Fig. 1: Illustration of mutual exclusivity.*

## References 
Image in figure 1: (https://louis.pressbooks.pub/finitemathematics/chapter/5-3-understanding-venn-diagrams/)