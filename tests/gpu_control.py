# GPU/accelerator control utilities for tests
# Keep this module lightweight: do not import heavy frameworks at module import time.

import os


def disable_gpu():
    """Force CPU-only execution across common frameworks.

    Must be called before importing TensorFlow, PyTorch, or JAX.
    """
    # Environment variables that most frameworks respect
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    # JAX prefers JAX_PLATFORMS since 0.4.x
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    # Reduce TensorFlow log noise
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    # Generic safety for other libs that might honor this
    os.environ.setdefault("OPENAI_ACCELERATE_DISABLE_CUDA", "1")

    # Try to disable TensorFlow GPUs via runtime API as an extra safeguard
    try:
        import tensorflow as tf  # Import only after env is set
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
    except Exception:
        # TensorFlow not available or failed to import; ignore
        pass
