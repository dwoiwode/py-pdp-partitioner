from typing import Union, Tuple, Iterable
import ConfigSpace.hyperparameters as CSH

ColorType = Union[str, Tuple[float, float, float]]
SelectedHyperparameterType = Union[CSH.Hyperparameter, str, Iterable[Union[str, CSH.Hyperparameter]]]