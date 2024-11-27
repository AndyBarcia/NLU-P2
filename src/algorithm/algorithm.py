from algorithm.sample import Sample
from algorithm.transition import Transition
from state import State
from conllu_token import Token


class ArcEager():

    """
    Implements the arc-eager transition-based parsing algorithm for dependency parsing.

    This class includes methods for creating initial parsing states, applying transitions to 
    these states, and determining the correct sequence of transitions for a given sentence.

    Methods:
        create_initial_state(sent: list[Token]): Creates the initial state for a given sentence.
        final_state(state: State): Checks if the current parsing state is a valid final configuration.
        LA_is_valid(state: State): Determines if a LEFT-ARC transition is valid for the current state.
        LA_is_correct(state: State): Determines if a LEFT-ARC transition is correct for the current state.
        RA_is_correct(state: State): Determines if a RIGHT-ARC transition is correct for the current state.
        RA_is_valid(state: State): Checks if a RIGHT-ARC transition is valid for the current state.
        REDUCE_is_correct(state: State): Determines if a REDUCE transition is correct for the current state.
        REDUCE_is_valid(state: State): Determines if a REDUCE transition is valid for the current state.
        oracle(sent: list[Token]): Computes the gold transitions for a given sentence.
        apply_transition(state: State, transition: Transition): Applies a given transition to the current state.
        gold_arcs(sent: list[Token]): Extracts gold-standard dependency arcs from a sentence.
    """

    LA = "LEFT-ARC"
    RA = "RIGHT-ARC"
    SHIFT = "SHIFT"
    REDUCE = "REDUCE"

    def create_initial_state(self, sent: list['Token']) -> State:
        """
        Creates the initial state for the arc-eager parsing algorithm given a sentence.

        This function initializes the parsing state, which is essential for beginning the parsing process. 
        The initial state consists of a stack (initially containing only the root token), a buffer 
        (containing all tokens of the sentence except the root), and an empty set of arcs.

        Parameters:
            sent (list[Token]): A list of 'Token' instances representing the sentence to be parsed. 
                                The first token in the list should typically be a 'ROOT' token.

        Returns:
            State: The initial parsing state, comprising a stack with the root token, a buffer with 
                the remaining tokens, and an empty set of arcs.
        """
        return State([sent[0]], sent[1:], set([]))
    
    def final_state(self, state: State) -> bool:
        """
        Checks if the curent parsing state is a valid final configuration, i.e., the buffer is empty

            Parameters:
                state (State): The parsing configuration to be checked

            Returns: A boolean that indicates if state is final or not
        """
        return len(state.B) == 0

    def LA_is_valid(self, state: State) -> bool:
        """
        Determines if a LEFT-ARC (LA) transition is valid for the current parsing state.

        A LEFT-ARC transition is valid if certain preconditions are met in the parser's state.
        This typically involves checking the current state of the stack and buffer in the parser.

        Parameters:
            state (State): The current state of the parser, including the stack and buffer.

        Returns:
            bool: True if a LEFT-ARC transition is valid in the current state, False otherwise.
        """

        # We can't apply LEFT-ARC if the token being processed is ROOT, as this would
        # leave the stack empty.
        dependent_token = state.S[-1]
        if dependent_token.form == "ROOT":
            assert len(state.S) == 1, "Stack should only contain ROOT"
            return False
        # LEFT-ARc would create an arc from the head_token (first token in the buffer)
        # to the dependent_token (last token in the stack). We need to check that this 
        # token isn't already the dependent of some existing arc.
        return dependent_token.id not in [dependent_id for (_, _, dependent_id) in state.A]

    def LA_is_correct(self, state: State) -> bool:
        """
        Determines if a LEFT-ARC (LA) transition is the correct action for the current parsing state.

        This method checks if applying a LEFT-ARC transition will correctly reflect the dependency
        structure of the sentence being parsed, based on the current state of the parser.

        Parameters:
            state (State): The current state of the parser, including the stack and buffer.

        Returns:
            bool: True if a LEFT-ARC transition is the correct action in the current state, False otherwise.
        """
        # First, check if this is a valid transition.
        if not self.LA_is_valid(state):
            return False
        # LEFT-ARc would create an arc from the head_token (first token in the buffer)
        # to the dependent_token (last token in the stack). We need to check if this
        # arc actually exists to determine if it is valid.
        head_token = state.B[0]
        dependent_token = state.S[-1]

        return dependent_token.head == head_token.id
    
    def RA_is_valid(self, state: State) -> bool:
        """
        Checks the preconditions in order to apply a right-arc (RA) transition.

        A RIGHT-ARC transition is valid under certain conditions related to the state of the stack
        and buffer in the parser. This method evaluates these conditions to determine if a RIGHT-ARC
        can be applied.

        Parameters:
            state (State): The current parsing state of the parser.

        Returns:
            bool: True if a RIGHT-ARC transition can be validly applied in the current state, False otherwise.
        """
        # RIGHT-ARc would create an arc from the head_token (last token in the stack)
        # to the dependent_token (first token in the buffer). We need to check that this 
        # token isn't already the dependent of some existing arc.
        dependent_token = state.B[0]
        return dependent_token.id not in [dependent_id for (_, _, dependent_id) in state.A]

    def RA_is_correct(self, state: State) -> bool:
        """
        Determines if a RIGHT-ARC (RA) transition is the correct action for the current parsing state.

        This method assesses whether applying a RIGHT-ARC transition aligns with the correct 
        dependency structure of the sentence, based on the parser's current state.

        Parameters:
            state (State): The current state of the parser, including the stack and buffer.

        Returns:
            bool: True if a RIGHT-ARC transition is the correct action in the current state, False otherwise.
        """    
        # First, check if this is a valid transition.
        if not self.RA_is_valid(state):
            return False
        # RIGHT-ARc would create an arc from the head_token (last token in the stack)
        # to the dependent_token (first token in the buffer). We need to check if this
        # arc actually exists to determine if it is valid.
        head_token = state.S[-1]
        dependent_token = state.B[0]

        return dependent_token.head == head_token.id

    def REDUCE_is_valid(self, state: State) -> bool:
        """
        Determines if a REDUCE transition is valid for the current parsing state.

        This method checks if the preconditions for applying a REDUCE transition are met in 
        the current state of the parser. This typically involves assessing the state of the 
        stack and buffer.

        Parameters:
            state (State): The current state of the parser, including the stack and buffer.

        Returns:
            bool: True if a REDUCE transition is valid in the current state, False otherwise.
        """
        # Reduce would pop that last token in the stack, so we need to make
        # sure that this token is the dependent in some arc before we remove it.
        dependent_token = state.S[-1]
        return dependent_token.id in [dependent_id for (_, _, dependent_id) in state.A]

    def REDUCE_is_correct(self, state: State) -> bool:
        """
        Determines if applying a REDUCE transition is the correct action for the current parsing state.

        A REDUCE transition is correct if there is no word in the buffer (state.B) whose head 
        is the word on the top of the stack (state.S[-1]). This method checks this condition 
        against the current state of the parser.

        REDUCE can be only correct iff for every word in the buffer, 
        no word has as head the top word from the stack 

        Parameters:
            state (State): The current state of the parser, including the stack and buffer.

        Returns:
            bool: True if a REDUCE transition is the correct action in the current state, False otherwise.
        """
        # First, check if this is a valid transition.
        if not self.REDUCE_is_valid(state):
            return False
        # It is correct to do REDUCE if there is no word in the buffer whose head is 
        # the word on the top of the stack (state.S[-1])
        head_token = state.S[-1]
        return all([ head_token.id != dependent_token.head for dependent_token in state.B ])

    def oracle(self, sent: list['Token']) -> list['Sample']:
        """
        Computes the gold transitions to take at each parsing step, given an input dependency tree.

        This function iterates through a given sentence, represented as a dependency tree, to generate a sequence 
        of gold-standard transitions. These transitions are what an ideal parser should predict at each step to 
        correctly parse the sentence. The function checks the validity and correctness of possible transitions 
        at each step and selects the appropriate one based on the arc-eager parsing algorithm. It is primarily 
        used for later training a dependency parser.

        Parameters:
            sent (list['Token']): A list of 'Token' instances representing a dependency tree. Each 'Token' 
                        should contain information about a word/token in a sentence.

        Returns:
            samples (list['Sample']): A list of Sample instances. Each Sample stores an state instance and a transition instance
            with the information of the outputs to predict (the transition and optionally the dependency label)
        """

        state = self.create_initial_state(sent) 

        samples = [] #Store here all training samples for sent

        #Applies the transition system until a final configuration state is reached
        while not self.final_state(state):
            
            if self.LA_is_valid(state) and self.LA_is_correct(state):
                #Add current state 'state' (the input) and the transition taken (the desired output) to the list of samples
                #Update the state by applying the LA transition using the function apply_transition
                raise NotImplementedError

            elif self.RA_is_valid(state) and self.RA_is_correct(state):
                #Add current state 'state' (the input) and the transition taken (the desired output) to the list of samples
                #Update the state by applying the RA transition using the function apply_transition
                raise NotImplementedError

            elif self.REDUCE_is_valid(state) and self.REDUCE_is_correct(state):
                #Add current state 'state' (the input) and the transition taken (the desired output) to the list of samples
                #Update the state by applying the REDUCE transition using the function apply_transition
                raise NotImplementedError
            else:
                #If no other transiton can be applied, it's a SHIFT transition
                transition = Transition(self.SHIFT)
                #Add current state 'state' (the input) and the transition taken (the desired output) to the list of samples
                samples.append(Sample(state, transition))
                #Update the state by applying the SHIFT transition using the function apply_transition
                self.apply_transition(state,transition)


        #When the oracle ends, the generated arcs must
        #match exactly the gold arcs, otherwise the obtained sequence of transitions is not correct
        assert self.gold_arcs(sent) == state.A, f"Gold arcs {self.gold_arcs(sent)} and generated arcs {state.A} do not match"
    
        return samples         
    

    def apply_transition(self, state: State, transition: Transition):
        """
        Applies a given transition to the current parsing state.

        This method updates the state based on the type of transition - LEFT-ARC, RIGHT-ARC, 
        or REDUCE - and the validity of applying such a transition in the current context.

        Parameters:
            state (State): The current parsing state, which includes a stack (S), 
                        a buffer (B), and a set of arcs (A).
            transition (Transition): The transition to be applied, consisting of an action
                                    (LEFT-ARC, RIGHT-ARC, REDUCE) and an optional dependency label (only for LEFT-ARC and RIGHT-arc).

        Returns:
            None; the state is modified in place.
        """

        # Extract the action and dependency label from the transition
        t = transition.action
        dep = transition.dependency

        # The top item on the stack and the first item in the buffer
        s = state.S[-1] if state.S else None  # Last in the stack
        b = state.B[0] if state.B else None   # First in the buffer

        if t == self.LA and self.LA_is_valid(state):
            # LEFT-ARc creates an arc from the head_token (first token in the buffer)
            # to the dependent_token (last token in the stack). The dependent_token
            # is removed from the top of the stack.
            state.A.append(
                # a --dep--> s
                (b,dep,s)
            )
            state.S.pop()

        elif t == self.RA and self.RA_is_valid(state): 
            # RIGHT-ARc creates an arc from the head_token (last token in the stack)
            # to the dependent_token (first token in the buffer). The dependent_token
            # is moved from the buffer to the top of the stack.
            state.A.append(
                # s --dep--> b
                (s,dep,b)
            )
            # Move from buffer to stack
            state.B.pop(0)
            state.S.push(b)

        elif t == self.REDUCE and self.has_head(s, state.A): 
            # REDUCE removes the word from the top of the stack.
            state.S.pop()
        else:
            # SHIFT transition logic: Already implemented! Use it as a basis to implement the others
            #This involves moving the top of the buffer to the stack
            state.S.append(b) 
            del state.B[:1]

    def gold_arcs(self, sent: list['Token']) -> set:
        """
        Extracts and returns the gold-standard dependency arcs from a given sentence.

        This function processes a sentence represented by a list of Token objects to extract the dependency relations 
        (arcs) present in the sentence. Each Token object should contain information about its head (the id of the 
        parent token in the dependency tree), the type of dependency, and its own id. The function constructs a set 
        of tuples, each representing a dependency arc in the sentence.

        Parameters:
            sent (list[Token]): A list of Token objects representing the sentence. Each Token object contains 
                                information about a word or punctuation in a sentence, including its dependency 
                                relations and other annotations.

        Returns:
            gold_arcs (set[tuple]): A set of tuples, where each tuple is a triplet (head_id, dependency_type, dependent_id). 
                                    This represents all the gold-standard dependency arcs in the sentence. The head_id and 
                                    dependent_id are integers representing the respective tokens in the sentence, and 
                                    dependency_type is a string indicating the type of dependency relation.
        """
        gold_arcs = set([])
        for token in sent[1:]:
            gold_arcs.add((token.head, token.dep, token.id))

        return gold_arcs


   


if __name__ == "__main__":


    print("**************************************************")
    print("*               Arc-eager function               *")
    print("**************************************************\n")

    print("Creating the initial state for the sentence: 'The cat is sleeping.' \n")

    tree = [
        Token(0, "ROOT", "ROOT", "_", "_", "_", "_", "_"),
        Token(1, "The", "the", "DET", "_", "Definite=Def|PronType=Art", 2, "det"),
        Token(2, "cat", "cat", "NOUN", "_", "Number=Sing", 4, "nsubj"),
        Token(3, "is", "be", "AUX", "_", "Mood=Ind|Tense=Pres|VerbForm=Fin", 4, "cop"),
        Token(4, "sleeping", "sleep", "VERB", "_", "VerbForm=Ger", 0, "root"),
        Token(5, ".", ".", "PUNCT", "_", "_", 4, "punct")
    ]

    arc_eager = ArcEager()
    print("Initial state")
    state = arc_eager.create_initial_state(tree)
    print(state)

    #Checking that is a final state
    print (f"Is the initial state a valid final state (buffer is empty)? {arc_eager.final_state(state)}\n")

    # Applying a SHIFT transition
    transition1 = Transition(arc_eager.SHIFT)
    arc_eager.apply_transition(state, transition1)
    print("State after applying the SHIFT transition:")
    print(state, "\n")

    #Obtaining the gold_arcs of the sentence with the function gold_arcs
    gold_arcs = arc_eager.gold_arcs(tree)
    print (f"Set of gold arcs: {gold_arcs}\n\n")


    print("**************************************************")
    print("*  Creating instances of the class Transition    *")
    print("**************************************************")

    # Creating a SHIFT transition
    shift_transition = Transition(ArcEager.SHIFT)
    # Printing the created transition
    print(f"Created Transition: {shift_transition}")  # Output: Created Transition: SHIFT

    # Creating a LEFT-ARC transition with a specific dependency type
    left_arc_transition = Transition(ArcEager.LA, "nsubj")
    # Printing the created transition
    print(f"Created Transition: {left_arc_transition}")

    # Creating a RIGHT-ARC transition with a specific dependency type
    right_arc_transition = Transition(ArcEager.RA, "amod")
    # Printing the created transition
    print(f"Created Transition: {right_arc_transition}")

    # Creating a REDUCE transition
    reduce_transition = Transition(ArcEager.REDUCE)
    # Printing the created transition
    print(f"Created Transition: {reduce_transition}")  # Output: Created Transition: SHIFT

    print()
    print("**************************************************")
    print("*     Creating instances of the class  Sample    *")
    print("**************************************************")

    # For demonstration, let's create a dummy State instance
    state = arc_eager.create_initial_state(tree)  # Replace with actual state initialization as per your implementation

    # Create a Transition instance. For example, a SHIFT transition
    shift_transition = Transition(ArcEager.SHIFT)

    # Now, create a Sample instance using the state and transition
    sample_instance = Sample(state, shift_transition)

    # To display the created Sample instance
    print("Sample:\n", sample_instance)