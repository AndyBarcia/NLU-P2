from typing import Any, List


def extract_last_n(array: list, n: int, pad_value: Any):
    """
    Extract the last `n` elements from a list. If the list has fewer than `n` elements,
    prepend the `pad_value` to make up the difference.

    Parameters:
        array (list): The input list to extract elements from.
        n (int): The number of elements to extract.
        pad_value (Any): The value to prepend if the list has fewer than `n` elements.

    Returns:
        list: A list containing the last `n` elements of the input list, padded if necessary.

    Example:
        >>> extract_last_n([1, 2, 3], 5, 0)
        [0, 0, 1, 2, 3]
        >>> extract_last_n([1, 2, 3], 2, 0)
        [2, 3]
    """
    # Ensure the result has at least n elements by prepending the pad_value if necessary
    padded_array = [pad_value] * max(0, n - len(array)) + array
    # Slice the last n elements
    return padded_array[-n:]

def extract_first_n(array: list, n: int, pad_value: Any):
    """
    Extract the first `n` elements from a list. If the list has fewer than `n` elements,
    append the `pad_value` to make up the difference.

    Parameters:
        array (list): The input list to extract elements from.
        n (int): The number of elements to extract.
        pad_value (Any): The value to append if the list has fewer than `n` elements.

    Returns:
        list: A list containing the first `n` elements of the input list, padded if necessary.

    Example:
        >>> extract_first_n([1, 2, 3], 5, 0)
        [1, 2, 3, 0, 0]
        >>> extract_first_n([1, 2, 3], 2, 0)
        [1, 2]
    """
    # Ensure the result has at least n elements by appending the pad_value if necessary
    padded_array = array + [pad_value] * max(0, n - len(array))
    # Slice the first n elements
    return padded_array[:n]

def flatten_list_of_lists(nested_list: List[list]):
    """
    Flatten a list of lists into a single list.

    Parameters:
        nested_list (list): A list containing other lists as elements. 

    Returns:
        list: A single flattened list containing all elements from the sublists.

    Example:
        >>> flatten_list_of_lists([[1, 2], [3, 4], [5]])
        [1, 2, 3, 4, 5]
    """
    return [item for sublist in nested_list for item in sublist]