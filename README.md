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

# Detailed Plane
## Dataset
Dataset should be well established. We should look at nlpprogress for idea.
I think I can later throw in the protein dataset. We must look if there is
already one. 

First we should select a small dataset. I think we should first focus on
language modeling, later we can investigate classification and translation.

For simplicity I think we should start with character level language model
datasets. From [nlpprogress](http://nlpprogress.com/english/language_modeling.html)
we have two options Hutter Prize and Text8. Text8 seems simpler lets start with that.
