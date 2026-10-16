"""Explicit, torch-free overrides for CPU contract checks and the GPU canary.

Production analysis never reads overrides from the output tree or environment.
The caller must pass this object (CLI: --test-contract-json); results are marked
contract_only and cannot be used for scientific labels.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TestContract:
    __test__ = False
    test_only: bool
    expected_steps: dict
    preset: str = 'mini-m'
    seeds: tuple = (3,)
    guard_every: int = 50
    n_docs: int = 1024
    n_ho_adapt: int = 128
    ho_docs: int = 256
    passkey_docs: int = 256
    grids: dict = field(default_factory=dict)
    passkey_etas: dict = field(default_factory=dict)
    passkey_hosts_file: str = ''
    passkey_digit_tokens: int = 2

    def __post_init__(self):
        if self.test_only is not True or not self.expected_steps:
            raise ValueError('explicit test_only=true and expected_steps are required')
        for n in [*self.expected_steps.values(), self.guard_every, self.n_docs,
                  self.n_ho_adapt, self.ho_docs, self.passkey_docs, self.passkey_digit_tokens]:
            if type(n) is not int or n < 1:
                raise ValueError('test contract counts/horizons must be positive integers')
