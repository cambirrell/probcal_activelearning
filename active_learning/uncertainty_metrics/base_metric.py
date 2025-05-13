class base_metric:
    """
    Base class for uncertainty metrics.
    """

    def __init__(self, name: str):
        self.name = name

    def compute_error(self, model, data):
        """
        Compute the error of the model on the data.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def compute_uncertainty(self, model, data):
        """
        Compute the uncertainty of the model on the data.
        """
        raise NotImplementedError("Subclasses should implement this method.")