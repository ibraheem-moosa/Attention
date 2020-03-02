# Goal
The goal of this project is threefold.
1. Explore the capabilities of the Ignite and thinc.ai framework.
2. Understand the attention mechanism.
3. Study the state of the art in language modeling.

# Guding Principle
1. We should strive to write as few lines of code as possible.
2. We should solve the problem at hand first and later do generalization.

# Roadmap
- [] Choose appropriate datasets.
- [] Implement DataLoader for the simplest dataset in Ignite.
- [] Implement a basic attention base model.
- [] Document results from experiment.

# Detailed Plan
## Dataset
Dataset should be well established. We should look at nlpprogress for idea.
We can later throw in a protein dataset.

First we should select a small dataset. I think we should first focus on
language modeling, later we can investigate classification and translation.

For simplicity I think we should start with character level language model
datasets. From [nlpprogress](http://nlpprogress.com/english/language_modeling.html)
we have two options Hutter Prize and Text8.

Text8 is a single text file with 10^8 characters.

## Ignite
We will first write a simple RNN model.

[Text8](http://mattmahoney.net/dc/text8.zip) seems simpler, lets start with that.

# Resouces
