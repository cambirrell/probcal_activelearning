from torch.utils.data import Dataset

class DatasetIndexMixin:
    """
    A Mixin for PyTorch Dataset classes to enable returning the sample index.

    When a class inherits from this Mixin, its __getitem__ method can be
    toggled to return (data, target, index) instead of just (data, target).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Ensures the next class in MRO is initialized
        self._return_index = False

    def return_index(self, value: bool = True):
        """
        Enable or disable returning the index in __getitem__.

        Args:
            value (bool): Set to True to return indices, False for normal behavior.
        """
        self._return_index = value

    def __getitem__(self, index):
        # This calls the __getitem__ of the class that inherits the Mixin
        # (e.g., EVADataset's implementation) by calling the next method
        # in Python's Method Resolution Order (MRO).
        item = super().__getitem__(index)

        if not self._return_index:
            return item
        else:
            # Assumes the base __getitem__ returns a tuple (e.g., (image, score))
            # and appends the index to it.
            return (*item, index)