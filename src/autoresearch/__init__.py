"""AutoResearch: discovery with a receipt.

A domain-general engine that searches a space of candidate hypotheses and emits
either a claim with a certified error rate, or a refusal naming the reason and
the sample size that would fix it.

The public surface is intentionally small:

    from autoresearch.contract import load_contract
    from autoresearch.power import assess
    from autoresearch.engine import Engine, RunConfig

Everything else is an implementation detail behind file-based schemas.
"""

__version__ = "0.1.0"

from .contract import Contract, ContractError, load_contract  # noqa: F401
from .power import Assessment, assess  # noqa: F401
from .engine import Engine, RunConfig  # noqa: F401
