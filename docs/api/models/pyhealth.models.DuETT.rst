pyhealth.models.DuETT
=====================

The Dual Event Time Transformer for sparse, irregularly sampled EHR time
series (`Labach et al., MLHC 2023
<https://proceedings.mlr.press/v219/labach23a.html>`_). Observations are
binned into a grid of time bins by event types, with the last value and the
observation count in each cell. Transformer layers alternate between
attending over event types and over time bins, and the model is pre-trained
to predict the values and presence of a masked time bin and a masked event
type before fine-tuning.

It reads the irregular series of
:class:`~pyhealth.tasks.PhysioNet2012Mortality` directly (and bins them
itself), or a task's pre-binned grid of values, counts and bin times.

.. autoclass:: pyhealth.models.DuETT
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.models.duett.bin_time_series

.. autoclass:: pyhealth.models.duett.DuETTEncoderLayer
    :members:
    :show-inheritance:

.. autoclass:: pyhealth.models.duett.PerVariableEmbedding
    :members:
    :show-inheritance:

.. autoclass:: pyhealth.models.duett.ScaleNorm
    :members:
    :show-inheritance:
