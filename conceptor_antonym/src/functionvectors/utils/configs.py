from dataclasses import dataclass
from enum import Enum


@dataclass
class ExperimentConfigConceptor:
    """
    Configuration for the Conceptor experiment

    Attributes:
        extraction_layer (int): The layer to extract activations from.
        beta (float): The beta value for conceptor steering.
        aperture (float): The aperture value for conceptors.
        lambda_ (float): Steering strength for the additive-projection update
            h' = h + lambda * (beta * C - I) h. 0.0 disables steering,
            1.0 applies the full update. Default 0.5.
    """
    extraction_layer: int
    beta: float
    aperture: float
    lambda_: float = 0.5


@dataclass
class ExperimentConfigAveraging:
    """
    Configuration for the Averaging experiment

    Attributes:
        extraction_layer (int): The layer to extract activations from.
        beta (float): The beta value for averaging.
    """
    extraction_layer: int
    beta: float


class ExperimentType(Enum):
    """
    Enum class for the different types of experiments
    """
    BASELINE = "baseline"
    CONCEPTOR = "conceptor"
    #: Pure published form, h' = beta_c * C * h (arXiv:2410.16314 eq. 4).
    #: Ignores lambda; equivalent to CONCEPTOR at lambda=1.
    CONCEPTOR_PAPER = "conceptor-paper"
    CONCEPTOR_MEAN = "conceptor-mean"
    ADDITION = "addition"
    ADDITION_MEAN = "addition-mean"