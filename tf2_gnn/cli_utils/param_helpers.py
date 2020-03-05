"""Functions to convert from string parameters to their values."""
import json
from typing import List, Dict, Any
from distutils.util import strtobool


def to_bool(val) -> bool:
    """ Accepts a boolean or str value, and returns the boolean equivalent, converting if necessary """

    if type(val) == bool:
        return val
    else:
        return bool(strtobool(val))


def str_to_list_of_ints(val) -> List[int]:
    """Accepts a str or list, returns list of ints. Specifically useful when 
    num_hidden_units of a set of layers is specified as a list of ints"""
    if type(val) == list:
        return val
    else:

        return [int(v) for v in json.loads(val)]


def override_model_params_with_hyperdrive_params(
    model_params: Dict[str, Any], hyperdrive_params: Dict[str, str]
):
    """
    Overrides the model parameters, with those from hyperdrive_params. hyperdrive_params contains hyperparameter values as strings. 
	The correct type is inferred from the model params (only the value is used from hyperdrive_params)
    """
    for k in hyperdrive_params.keys():
        if k not in model_params:
            raise ValueError(f"key {k} not found in model_params: {model_params}")

        if type(model_params[k]) == bool:
            model_params[k] = to_bool(hyperdrive_params[k])
        elif type(model_params[k]) == int:
            model_params[k] = int(hyperdrive_params[k])
        elif type(model_params[k]) == float:
            model_params[k] = float(hyperdrive_params[k])
        elif type(model_params[k]) == list and type(model_params[k][0]) == int:
            model_params[k] = str_to_list_of_ints(hyperdrive_params[k])
        elif type(model_params[k]) == str:
            model_params[k] = hyperdrive_params[k]
        else:
            raise ValueError(f"Unknown hyperparameter type {type(model_params[k])} for hyperparameter {k}.")
    return
