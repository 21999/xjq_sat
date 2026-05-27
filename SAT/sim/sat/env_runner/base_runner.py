from typing import Dict
from sat.policy.base_policy import BasePolicy


class BaseRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BasePolicy, vqvae) -> Dict:
        raise NotImplementedError()
