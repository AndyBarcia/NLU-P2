from src.conllu_reader import ConlluReader
from src.algorithm.algorithm import ArcEager
from src.model import ParserMLP

from tqdm import tqdm
from typing import Dict

def read_file(reader, path, inference):
    trees = reader.read_conllu_file(path, inference)
    print(f"Read a total of {len(trees)} sentences from {path}")
    #print (f"Printing the first sentence of the training set... trees[0] = {trees[0]}")
    #for token in trees[0]:
    #    print (token)
    #print()
    return trees


# Initialize the ConlluReader
reader = ConlluReader()

print ("\n ------ Loading CoNLLU files ------")

train_trees = read_file(reader,path="conllu/en_partut-ud-train_clean.conllu", inference=False)
dev_trees = read_file(reader,path="conllu/en_partut-ud-dev_clean.conllu", inference=False)
test_trees = read_file(reader,path="conllu/en_partut-ud-test_clean.conllu", inference=True)

"""
We remove the non-projective sentences from the training and development set,
as the Arc-Eager algorithm cannot parse non-projective sentences.

We don't remove them from test set set, because for those we only will do inference
"""

print ("\n ------ Removing Non Projective Trees ------")

train_trees = reader.remove_non_projective_trees(train_trees)
dev_trees = reader.remove_non_projective_trees(dev_trees)

print ("Total training trees after removing non-projective sentences", len(train_trees))
print ("Total dev trees after removing non-projective sentences", len(dev_trees))

#Create and instance of the ArcEager
arc_eager = ArcEager()

print ("\n ------ Generating Samples from Sentence Tokens ------")

train_trees_samples = [ arc_eager.oracle(tokens) for tokens in tqdm(train_trees, desc="Train") ]
dev_trees_samples   = [ arc_eager.oracle(tokens) for tokens in tqdm(dev_trees, desc="Dev") ]

print ("\n ------ Training model ------")

model = ParserMLP(epochs=1)
model.train(train_trees_samples, dev_trees_samples)
model.evaluate(dev_trees_samples)

model.run(test_trees)

# TODO: Define and implement the neural model in the 'model.py' module.
# 1. Train the model on the generated training dataset.
# 2. Evaluate the model's performance using the development dataset.
# 3. Conduct inference on the test set with the trained model.
# 4. Save the parsing results of the test set in CoNLLU format for further analysis.

# TODO: Utilize the 'postprocessor' module (already implemented).
# 1. Read the output saved in the CoNLLU file and address any issues with ill-formed trees.
# 2. Specify the file path: path = "<YOUR_PATH_TO_OUTPUT_FILE>"
# 3. Process the file: trees = postprocessor.postprocess(path)
# 4. Save the processed trees to a new output file.