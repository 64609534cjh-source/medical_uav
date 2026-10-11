import numpy as np
import pytest

from medical_uav.env.config import EnvConfig
from medical_uav.env.map_gen import CityMap
from medical_uav.env.orders import Order
from medical_uav.env.simulator import Simulator

# Nodes on a line, distances are |x_i - x_j| km: M, H1..H5, C2
LINE_X = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.5]


@pytest.fixture
def line_map():
    return CityMap.from_positions(np.array([[x, 0.0] for x in LINE_X]))


def make_order(oid, type_idx, origin, dest, weight, release):
    limits = [30.0, 90.0, 60.0, 75.0]
    limit = limits[type_idx]
    return Order(oid, type_idx, origin, dest, weight, release, limit, 2 * limit)


def make_sim(city_map, orders, cfg=None):
    return Simulator(cfg or EnvConfig(), city_map, orders, lam=0.07)
