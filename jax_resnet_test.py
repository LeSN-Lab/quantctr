import jax.numpy as jnp
from jax_resnet import pretrained_resnest

ResNeSt50, variables = pretrained_resnest(50)
model = ResNeSt50()
out = model.apply(variables,
                  jnp.ones((32, 224, 224, 3)),  # ImageNet sized inputs.
                  mutable=False)  # Ensure `batch_stats` aren't updated.