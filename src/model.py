from src.conllu_token import Token
from src.algorithm.sample import Sample
from src.utils import flatten_list_of_lists
from src.id_mapping import MappingStrID

import tensorflow as tf

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
    
        self.word_mapping = MappingStrID(include_padding=True)
        self.transition_mapping = MappingStrID()    
    
    def _samples_to_features_and_targets(self, samples: list['Sample']):
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

        # Get targets of the sample set: the transition to be taken at each sample.
        trees_targets = flatten_list_of_lists(
            [ sample.transition for sample in samples ]
            for samples in samples
        ) # [ transition, transition... ]

        # Convert string targets to int IDs.
        trees_targets = tf.constant([
            self.transition_mapping.register_obj(targets) 
            for targets in trees_targets 
        ]) # shape (N,)

        return trees_feats, trees_targets

    def train(self, training_samples: list['Sample'], dev_samples: list['Sample']):
        """
        Trains the MLP model using the provided training and development samples.

        This method prepares the training data by mapping samples to IDs suitable for 
        embedding layers and then proceeds to compile and fit the Keras model.

        Parameters:
            training_samples (list[Sample]): A list of training samples for the parser.
            dev_samples (list[Sample]): A list of development samples used for model validation.
        """

        # Convert samples to features and targets.
        train_trees_feats, train_trees_targets = self._samples_to_features_and_targets(training_samples)
        dev_trees_feats, dev_trees_targets = self._samples_to_features_and_targets(dev_samples)

        

        raise NotImplementedError

    def evaluate(self, samples: list['Sample']):
        """
        Evaluates the model's performance on a set of samples.

        This method is used to assess the accuracy of the model in predicting the correct
        transition and dependency types. The expected accuracy range is between 75% and 85%.

        Parameters:
            samples (list[Sample]): A list of samples to evaluate the model's performance.
        """
        raise NotImplementedError
    
    def run(self, sents: list['Token']):
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