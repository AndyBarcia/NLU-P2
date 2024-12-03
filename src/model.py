from src.conllu_token import Token
from src.algorithm.sample import Sample
from src.utils import flatten_list_of_lists
from src.id_mapping import MappingStrID
from src.metrics import SparseCategoricalAccuracyIgnoreClass
from src.algorithm.algorithm import ArcEager
from src.algorithm.sample import Sample

import tensorflow as tf
from keras import layers, models
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy

from typing import List

class ParserMLP:
    """
    A Multi-Layer Perceptron (MLP) class for a dependency parser, using TensorFlow and Keras.

    This class implements a neural network model designed to predict transitions in a dependency 
    parser. It utilizes the Keras Functional API, which is more suited for multi-task learning scenarios 
    like this one. The network is trained to map parsing states to transition actions, facilitating 
    the parsing process in natural language processing tasks.

    Attributes:
        word_emb_dim (int): Dimensionality of the word embeddings. Defaults to 100.
        hidden_dim (int): Dimension of the hidden layer in the neural network. Defaults to 64.
        epochs (int): Number of training epochs. Defaults to 1.
        batch_size (int): Size of the batches used in training. Defaults to 64.

    Methods:
        train(training_samples, dev_samples): Trains the MLP model using the provided training and 
            development samples. It maps these samples to IDs that can be processed by an embedding 
            layer and then calls the Keras compile and fit functions.

        evaluate(samples): Evaluates the performance of the model on a given set of samples. The 
            method aims to assess the accuracy in predicting both the transition and dependency types, 
            with expected accuracies ranging between 75% and 85%.

        run(sents): Processes a list of sentences (tokens) using the trained model to perform dependency 
            parsing. This method implements the vertical processing of sentences to predict parser 
            transitions for each token.

        Feel free to add other parameters and functions you might need to create your model
    """

    def __init__(self, word_emb_dim: int = 100, hidden_dim: int = 64, 
                 epochs: int = 1, batch_size: int = 64):
        """
        Initializes the ParserMLP class with the specified dimensions and training parameters.

        Parameters:
            word_emb_dim (int): The dimensionality of the word embeddings.
            hidden_dim (int): The size of the hidden layer in the MLP.
            epochs (int): The number of epochs for training the model.
            batch_size (int): The batch size used during model training.
        """
        
        self.nbuffer_feats = 5
        self.nstack_feats = 5
        self.input_dim = 2 * (self.nbuffer_feats + self.nstack_feats)
        self.word_emb_dim = word_emb_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size

        self.word_mapping = MappingStrID(include_padding=True)
        self.action_mapping = MappingStrID()
        self.dependency_mapping = MappingStrID()
    
    def _samples_to_features_and_targets(self, samples: List[List[Sample]]):
        # Get features of the sample set based on N buffer elements and M stack elements.
        trees_feats = flatten_list_of_lists([
            [ sample.state_to_feats(self.nbuffer_feats, self.nstack_feats) for sample in samples ] 
            for samples in samples 
        ]) # [[ str, str, str... ]]

        # Convert string features to int IDs.
        trees_feats = tf.constant([ 
            [ self.word_mapping.register_obj(feats) for feats in sentence_feats ] 
            for sentence_feats in trees_feats 
        ]) # shape (N, 2*(nbuffer_feats+nstack_feats))

        # Get action targets of the sample set: the transition to be taken at each sample.
        trees_action_targets = flatten_list_of_lists(
            [ sample.transition.action for sample in samples if sample.transition ]
            for samples in samples
        ) # [ transition, transition... ]

        # Convert string action targets to int IDs.
        trees_action_targets = tf.constant([
            self.action_mapping.register_obj(action_target) 
            for action_target in trees_action_targets 
        ]) # shape (N,)

        # Get dependency targets of the sample set: the transition to be taken at each sample.
        trees_dependency_targets = flatten_list_of_lists(
            [ sample.transition.dependency for sample in samples if sample.transition ]
            for samples in samples
        ) # [ dependency, dependency... ]

        # Convert string targets to int IDs. Note that the (None) dependency target
        # that corresponds to actions with no dependencies (REDUCE and SHIFT) will
        # also be converted to an ID.
        trees_dependency_targets = tf.constant([
            self.dependency_mapping.register_obj(dependency_target) 
            for dependency_target in trees_dependency_targets 
        ]) # shape (N,)

        return trees_feats, trees_action_targets, trees_dependency_targets

    def train(self, training_samples: List[List[Sample]], dev_samples: List[List[Sample]]):
        """
        Trains the MLP model using the provided training and development samples.

        This method prepares the training data by mapping samples to IDs suitable for 
        embedding layers and then proceeds to compile and fit the Keras model.

        Parameters:
            training_samples (List[List[Sample]]): A batched list of training samples for the parser.
            dev_samples (List[List[Sample]]): A batched list of development samples used for model validation.
        """

        # Convert samples to features and targets.
        train_x, train_action_y, train_dep_y = self._samples_to_features_and_targets(training_samples)
        dev_x, dev_action_y, dev_dep_y = self._samples_to_features_and_targets(dev_samples)

        # Get the ID assigned to the dependency of the SHIFT and REDUCE actions.
        # This actions have no dependencies.
        null_dependency_id = self.dependency_mapping.obj_to_id(None)

        self.num_words = len(self.word_mapping)
        self.num_actions = len(self.action_mapping)
        self.num_dependencies = len(self.dependency_mapping)

        # Loads matrix of labels of size (N, 2*(nbuffer_feats+nstack_feats))
        input_layer = layers.Input(shape=(self.input_dim,), dtype='int32') 
        # Convert labels to features (N, 2*(nbuffer_feats+nstack_feats), word_emb_dim)
        embedding_layer = layers.Embedding(
            input_dim=self.num_words + 1,
            output_dim=self.word_emb_dim
        )(input_layer)
        # Flatten features of the words into a single vector of size 
        # (N, 2*(nbuffer_feats+nstack_feats)*word_emb_dim)
        flattened_embedding = layers.Flatten()(embedding_layer)
        # Process in MLP, with output of shape (N, hidden_dim)
        output = layers.Dense(self.hidden_dim, activation='relu')(flattened_embedding)

        # Get the classification logits of the action to take,
        # Of shape (N,4) with labels corresponding to LEFT-ARC,
        # RIGHT-ARC, SHIFT or REDUCE
        logits_actions = layers.Dense(self.num_actions)(output)

        # Get the classification logits of the dependency of the
        # action to take. Shape (N,num_dependencies).
        logits_dependencies = layers.Dense(self.num_dependencies)(output) 

        # Define the model
        self.model = models.Model(inputs=input_layer, outputs={
            "action_pred": logits_actions, # (N,num_actions)
            "dependency_pred": logits_dependencies # (N,num_dependencies)
        })

        # Print model summary
        self.model.summary()
        
        # Compile the model. Remember to use ignore_class for both the loss and the
        # metrics so that the actions REDUCE and SHIFT don't penalize the classification
        # head for the depedency.
        self.model.compile(
            optimizer='adam',
            loss={
                'action_pred': SparseCategoricalCrossentropy(from_logits=True),
                'dependency_pred': SparseCategoricalCrossentropy(
                    from_logits = True,
                    ignore_class = null_dependency_id
                ),
            },
            metrics={
                'action_pred': SparseCategoricalAccuracy(),
                'dependency_pred': SparseCategoricalAccuracyIgnoreClass(ignore_class = null_dependency_id)
            }
        )

        # Train the model.
        self.model.fit(train_x, {
            'action_pred': train_action_y, 
            'dependency_pred': train_dep_y
        }, validation_data=(
            dev_x, {
                'action_pred': dev_action_y, 
                'dependency_pred': dev_dep_y
        }), epochs=self.epochs)

        return self.model

    def evaluate(self, samples: List[List[Sample]]):
        """
        Evaluates the model's performance on a set of samples.

        This method is used to assess the accuracy of the model in predicting the correct
        transition and dependency types. The expected accuracy range is between 75% and 85%.

        Parameters:
            samples (list[Sample]): A list of samples to evaluate the model's performance.
        """

        # Convert samples to features and targets.
        x, action_y, dep_y = self._samples_to_features_and_targets(samples)

        # Evaluate the model
        self.model.evaluate(x, {
            'action_pred': action_y, 
            'dependency_pred': dep_y
        })
    
    def run(self, sents: List[List[Token]]):
        """
        Executes the model on a list of sentences to perform dependency parsing.

        This method implements the vertical processing of sentences, predicting parser 
        transitions for each token in the sentences.

        Parameters:
            sents (list[Token]): A list of sentences, where each sentence is represented 
                                 as a list of Token objects.
        """

        # Main Steps for Processing Sentences:
        # 1. Initialize: Create the initial state for each sentence.
        # 2. Feature Representation: Convert states to their corresponding list of features.
        # 3. Model Prediction: Use the model to predict the next transition and dependency type for all current states.
        # 4. Transition Sorting: For each prediction, sort the transitions by likelihood using numpy.argsort, 
        #    and select the most likely dependency type with argmax.
        # 5. Validation Check: Verify if the selected transition is valid for each prediction. If not, select the next most likely one.
        # 6. State Update: Apply the selected actions to update all states, and create a list of new states.
        # 7. Final State Check: Remove sentences that have reached a final state.
        # 8. Iterative Process: Repeat steps 2 to 7 until all sentences have reached their final state.


        raise NotImplementedError


if __name__ == "__main__":
    
    model = ParserMLP()