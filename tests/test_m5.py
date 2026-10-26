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


def test_predeclared_lr_schedule_boundaries():
    from medical_uav.training.trainer import learning_rate_at
    cfg=TrainConfig(total_steps=1024000,learning_rate_schedule='late_linear_10pct')
    for step in (1024,102400,512000):assert learning_rate_at(cfg,step)==3e-4
    assert learning_rate_at(cfg,513024)==pytest.approx(.00029946)
    assert learning_rate_at(cfg,768000)==pytest.approx(.000165)
    assert learning_rate_at(cfg,1024000)==pytest.approx(.00003)
    assert learning_rate_at(TrainConfig(),1024000)==3e-4


def test_exact_kl_diagnostics_preserve_original_optimizer_update():
    import copy
    from pathlib import Path
    from medical_uav.algorithms.ppo import PPOConfig,PPOUpdater
    from medical_uav.training.collector import Collector
    torch.manual_seed(7);torch.set_num_threads(1)
    model=build_network('structured');original=copy.deepcopy(model)
    code=(Path(__file__).parent/'fixtures/m4_ppo.py').read_text()
    namespace={};exec(compile(code,'m4_ppo.py','exec'),namespace)
    cfg=PPOConfig();old=namespace['PPOUpdater'](original,namespace['PPOConfig'](),7)
    collector=Collector(8,7,('low','medium','high'))
    buffer,_=collector.collect(model,128);collector.close();buffer.finish()
    before={k:v.clone() for k,v in model.state_dict().items()}
    new_stats=PPOUpdater(model,cfg,7).update(buffer);old_stats=old.update(buffer)
    assert all(torch.equal(value,original.state_dict()[k]) for k,value in model.state_dict().items())
    for key in old_stats:assert new_stats[key]==old_stats[key]
    assert new_stats['exact_categorical_kl_mean']>=-1e-6
    assert new_stats['old_distribution_float32_bytes']==106496
    with torch.no_grad():
        masks=buffer.action_mask.flatten(0,1);obs=buffer.obs.flatten(0,1)
        prior=build_network('structured');prior.load_state_dict(before)
        from medical_uav.agents.networks import masked_distribution
        a=masked_distribution(prior(obs)[0],masks);b=masked_distribution(model(obs)[0],masks)
        exact=torch.distributions.kl_divergence(a,b)
    assert new_stats['exact_categorical_kl_mean']==pytest.approx(float(exact.mean()),abs=1e-6)
