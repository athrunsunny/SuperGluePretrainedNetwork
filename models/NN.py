import torch
import inspect
import sys
from abc import ABCMeta, abstractmethod
from copy import copy

from torch import nn
# from ..utils.base_model import BaseModel

class BaseModel(nn.Module, metaclass=ABCMeta):
    default_conf = {}
    required_inputs = []

    def __init__(self, conf):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        self.conf = conf = {**self.default_conf, **conf}
        self.required_inputs = copy(self.required_inputs)
        self._init(conf)
        sys.stdout.flush()

    def forward(self, data):
        """Check the data and call the _forward method of the child model."""
        for key in self.required_inputs:
            assert key in data, "Missing key {} in data".format(key)
        return self._forward(data)

    @abstractmethod
    def _init(self, conf):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def _forward(self, data):
        """To be implemented by the child class."""
        raise NotImplementedError

def find_nn(sim, ratio_thresh, distance_thresh):
    sim_nn, ind_nn = sim.topk(2 if ratio_thresh else 1, dim=-1, largest=True)
    dist_nn = 2 * (1 - sim_nn)
    mask = torch.ones(ind_nn.shape[:-1], dtype=torch.bool, device=sim.device)
    if ratio_thresh:
        mask = mask & (dist_nn[..., 0] <= (ratio_thresh**2) * dist_nn[..., 1])
    if distance_thresh:
        mask = mask & (dist_nn[..., 0] <= distance_thresh**2)
    matches = torch.where(mask, ind_nn[..., 0], ind_nn.new_tensor(-1))
    scores = torch.where(mask, (sim_nn[..., 0] + 1) / 2, sim_nn.new_tensor(0))
    return matches, scores


def mutual_check(m0, m1):
    inds0 = torch.arange(m0.shape[-1], device=m0.device)
    loop = torch.gather(m1, -1, torch.where(m0 > -1, m0, m0.new_tensor(0)))
    ok = (m0 > -1) & (inds0 == loop)
    m0_new = torch.where(ok, m0, m0.new_tensor(-1))
    return m0_new


class NearestNeighbor(BaseModel):
    default_conf = {
        "ratio_threshold": None,
        "distance_threshold": None,
        "do_mutual_check": True,
    }
    required_inputs = ["descriptors0", "descriptors1"]

    def _init(self, conf):
        pass

    def _forward(self, data):
        if data["descriptors0"].size(-1) == 0 or data["descriptors1"].size(-1) == 0:
            matches0 = torch.full(
                data["descriptors0"].shape[:2], -1, device=data["descriptors0"].device
            )
            return {
                "matches0": matches0,
                "matching_scores0": torch.zeros_like(matches0),
            }
        ratio_threshold = self.conf["ratio_threshold"]
        if data["descriptors0"].size(-1) == 1 or data["descriptors1"].size(-1) == 1:
            ratio_threshold = None
        sim = torch.einsum("bdn,bdm->bnm", data["descriptors0"], data["descriptors1"])
        matches0, scores0 = find_nn(
            sim, ratio_threshold, self.conf["distance_threshold"]
        )
        if self.conf["do_mutual_check"]:
            matches1, scores1 = find_nn(
                sim.transpose(1, 2), ratio_threshold, self.conf["distance_threshold"]
            )
            matches0 = mutual_check(matches0, matches1)
        return {
            "matches0": matches0,
            "matching_scores0": scores0,
        }
