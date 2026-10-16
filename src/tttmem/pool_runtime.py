"""POOL-owned budget and attempt bookkeeping, separate from scientific outcomes."""
from pathlib import Path
import json

from . import frs_state as state


class BudgetExhausted(Exception):
    """A resumable session boundary; never an infrastructure retry."""


def budget_record(**identity):
    return dict(identity, status='MISSING', disposition='TRUNCATED',
                reason='budget exhausted', resumable=True, retryable=True)


def mark_budget(folder, attempt):
    folder = Path(folder)
    state.atomic_json(folder/'budget.json', dict(resumable=True, attempt=attempt))


def saved_attempt(folder, default=1):
    file = Path(folder)/'attempt.json'
    return json.loads(file.read_text())['attempt'] if file.is_file() else default


def archive_budget(folder):
    """Retain cut evidence without counting it as the infrastructure retry."""
    folder = Path(folder)
    n = 1
    while folder.with_name(folder.name+f'.budget-{n}').exists():
        n += 1
    folder.rename(folder.with_name(folder.name+f'.budget-{n}'))
