"""Split isolation and additive diagnostic equivalence before M5 evaluation."""

import pytest
import torch

from medical_uav.agents.networks import build_network
from medical_uav.evaluation.evaluate import evaluate, selection_score, split_seeds, validate_days
from medical_uav.training.trainer import TrainConfig


def test_extra_is_development_only_and_cannot_select_checkpoints():
    assert split_seeds('dev-extra') == tuple(range(100040,100140))
    assert not set(split_seeds('dev-extra')) & set(split_seeds('dev'))
    assert not set(split_seeds('dev-extra')) & set(split_seeds('test'))
    with pytest.raises(ValueError): validate_days((100040,), 'dev')
    with pytest.raises(ValueError): TrainConfig(eval_split='dev-extra').validate()
    with pytest.raises(ValueError): selection_score({'policies':{'ppo':{'by_load':{}}},'split':'dev-extra'})


def test_diagnostics_do_not_change_complete_day_results():
    torch.manual_seed(4);torch.set_num_threads(1)
    model=build_network('structured')
    args=dict(network=model,policies=('ppo','slack'),seeds=(100002,),split='dev',loads=('medium','high'))
    plain,_=evaluate(**args);observed,_=evaluate(**args,diagnostics=True)
    for a,b in zip(plain,observed):
        extra=b.pop('diagnostics');assert a==b
        n=extra['outcomes'];assert n['released']==a['released']
        assert n['expired_unpicked']+n['expired_after_pick']==a['expired']
        assert extra['delivery']['delivered_boxes']==a['delivered']
        assert extra['resources']['departure_battery']['n']==extra['resources']['departure_payload_kg']['n']
