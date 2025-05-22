import jax

# NB: this is required to achieve reproducible results, especially on different
# CUDA versions/GPUs
# TODO: determine impact on training/inference speed
jax.config.update("jax_default_matmul_precision", "highest")
