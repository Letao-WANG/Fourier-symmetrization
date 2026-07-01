"""General Torch operators used to reproduce PennyLane-style circuits.

Use one import in notebooks:

```
import qmps as qmp
```

The package exposes reusable state, operation, and measurement primitives.
"""

from . import math
from . import operation
from . import qpinn
from . import statevector
from .operation import (
    CNOT,
    QSP,
    QState,
    Rot,
    RX,
    RY,
    RZ,
    RXFromZ,
    State,
    Hadamard,
    IsingZZ,
    PauliX,
    PauliZ,
    StronglyEntanglingLayers,
    X,
    ZeroState,
    ctrl,
    expval,
    expval_z,
)
from .math import as_scalar, prepare_broadcast_batch, to_reference_tensor
from .qpinn import (
    QFMConfig,
    ReuploadingStrongEntangleQFM,
    boundary_envelope,
    count_trainable_parameters,
    group_transformed_batch,
    hyperoctahedral_transformed_batch,
    make_qfm_models,
    map_circuit_input,
    sample_hyperoctahedral_indices,
)


def clear_caches() -> None:
    """Clear cached index/sign tensors used by the operator modules."""

    statevector.clear_caches()


__all__ = [
    "as_scalar",
    "clear_caches",
    "CNOT",
    "ctrl",
    "expval",
    "expval_z",
    "Hadamard",
    "IsingZZ",
    "math",
    "operation",
    "PauliX",
    "PauliZ",
    "prepare_broadcast_batch",
    "qpinn",
    "QFMConfig",
    "QSP",
    "QState",
    "boundary_envelope",
    "count_trainable_parameters",
    "group_transformed_batch",
    "hyperoctahedral_transformed_batch",
    "make_qfm_models",
    "map_circuit_input",
    "Rot",
    "RX",
    "RY",
    "RZ",
    "RXFromZ",
    "ReuploadingStrongEntangleQFM",
    "sample_hyperoctahedral_indices",
    "State",
    "statevector",
    "StronglyEntanglingLayers",
    "to_reference_tensor",
    "X",
    "ZeroState",
]
