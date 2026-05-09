from __future__ import annotations

import os
import random
import warnings


def setup_runtime(seed: int = 42) -> None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["PYTHONHASHSEED"] = str(seed)
    warnings.filterwarnings("ignore")

    import numpy as np
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.get_logger().setLevel("ERROR")
    tf.autograph.set_verbosity(0)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
