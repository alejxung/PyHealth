pyhealth.tasks.DuETTMortalityMIMIC4
===================================

MIMIC-IV in-hospital mortality from the first 48 hours of an ICU stay, in the
cohort of DuETT (`Labach et al., MLHC 2023
<https://proceedings.mlr.press/v219/labach23a.html>`_): 114 chart and lab
variables binned into a grid of last values and observation counts, plus the
paper's static variables. Made for :class:`~pyhealth.models.DuETT`.

.. autoclass:: pyhealth.tasks.DuETTMortalityMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.fit_value_normalizer

.. autofunction:: pyhealth.tasks.normalize_values
