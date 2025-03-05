""" """
import random 


def shuffle_datadict(dataset):
    """Shuffle a dictionary by key."""
    items = list(dataset.items())
    random.shuffle(items)
    return dict(items)