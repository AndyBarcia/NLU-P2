from src.algorithm.transition import Transition
from src.state import State
from src.utils import extract_first_n, extract_last_n, flatten_list_of_lists

from typing import Optional

class Sample(object):
    """
    Represents a training sample for a transition-based dependency parser. 

    This class encapsulates a parser state and the corresponding transition action 
    to be taken in that state. It is used for training models that predict parser actions 
    based on the current state of the parsing process.

    Attributes:
        state (State): An instance of the State class, representing the current parsing 
                       state at a given timestep in the parsing process.
        transition (Transition): An instance of the Transition class, representing the 
                       parser action to be taken in the given state. It can be None if
                       the transition to be taken is still unknown (the model is running
                       in inference mode).

    Methods:
        state_to_feats(nbuffer_feats: int = 2, nstack_feats: int = 2): Extracts features from the parsing state.
    """

    def __init__(self, state: State, transition: Optional[Transition] = None):
        """
        Initializes a new instance of the Sample class.

        Parameters:
            state (State): The current parsing state.
            transition (Transition): The transition action corresponding to the state.
        """
        self._state = state
        self._transition = transition

    @property
    def state(self):
        """
        Retrieves the current parsing state of the sample.

        Returns:
            State: The current parsing state in this sample.
        """
        return self._state


    @property
    def transition(self):
        """
        Retrieves the transition action of the sample.

        Returns:
            Transition: The transition action representing the parser's decision at this sample's state.
        """
        return self._transition


    def state_to_feats(self, nbuffer_feats: int = 2, nstack_feats: int = 2):
        """
        Extracts features from a given parsing state for use in a transition-based dependency parser.

        This function generates a feature representation from the current state of the parser, 
        which includes features from both the stack and the buffer. The number of features from 
        the stack and the buffer can be specified.

        Parameters:
            nbuffer_feats (int): The number of features to extract from the buffer.
            nstack_feats (int): The number of features to extract from the stack.

        Returns:
            list[str]: A list of extracted features. The features include the words and their 
                    corresponding UPOS (Universal Part-of-Speech) tags from the specified number 
                    of elements in the stack and buffer. The format of the feature list is as follows:
                    [Word_stack_n,...,Word_stack_0, Word_buffer_0,...,Word_buffer_m, 
                        UPOS_stack_n,...,UPOS_stack_0, UPOS_buffer_0,...,UPOS_buffer_m]
                    where 'n' is nstack_feats and 'm' is nbuffer_feats.

        Examples:
            Example 1:
                State: Stack (size=1): (0, ROOT, ROOT_UPOS)
                    Buffer (size=13): (1, Distribution, NOUN) | ... | (13, ., PUNCT)
                    Arcs (size=0): []

                Output: ['<PAD>', 'ROOT', 'Distribution', 'of', '<PAD>', 'ROOT_UPOS', 'NOUN', 'ADP']

            Example 2:
                State: Stack (size=2): (0, ROOT, ROOT_UPOS) | (1, Distribution, NOUN)
                    Buffer (size=10): (4, license, NOUN) | ... | (13, ., PUNCT)
                    Arcs (size=2): [(4, 'det', 3), (4, 'case', 2)]

                Output: ['ROOT', 'Distribution', 'license', 'does', 'ROOT_UPOS', 'NOUN', 'NOUN', 'AUX']
        """

        # Get the N first tokens from the buffer with padding included.
        buffer_tokens = extract_first_n(self.state.B, nbuffer_feats, pad_value=None)
        # Get the M last tokens from the stack
        stack_tokens = extract_last_n(self.state.S, nstack_feats, pad_value=None)

        # Get the actual words from the tokens.
        buffer_form = [tok.form if tok else '<PAD>' for tok in buffer_tokens]
        buffer_upos = [tok.upos if tok else '<PAD>' for tok in buffer_tokens]
        # Get the UPOS tags from the tokens.
        stack_form = [tok.form if tok else '<PAD>' for tok in stack_tokens]
        stack_upos = [tok.upos if tok else '<PAD>' for tok in stack_tokens]

        # Join the 4 lists together.
        return flatten_list_of_lists([
            stack_form, buffer_form,
            stack_upos, buffer_upos
        ])

    def __str__(self):
        """
        Returns a string representation of the sample, including its state and transition.

        Returns:
            str: A string representing the state and transition of the sample.
        """
        return f"Sample - State:\n\n{self._state}\nSample - Transition: {self._transition}"