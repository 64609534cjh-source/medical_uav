"""Physical time units, two isolated hypotheses, same information and real PPO."""
import json
import numpy as np
import pytest
import torch
from medical_uav.agents.networks import build_network,masked_distribution
from medical_uav.agents.decision import urgency_features,state_and_destinations,URGENCY_NAMES
from medical_uav.agents.destination_inference import DestinationInferencePolicy,factored_forward
from medical_uav.env.config import ORDER_TYPES
from medical_uav.training.trainer import TrainConfig,train,load_checkpoint
from medical_uav.evaluation.type_stats import type_bootstrap
from tests.test_structured import real_state


def test_physical_time_not_cross_type_normalized_minimum_and_expired_capacity():
    cargo=torch.zeros(3,16);cargo[:,0]=1;cargo[:,7]=1 # destination 2
    cargo[0,2]=1;cargo[0,11]=.3;cargo[0,12]=9/90;cargo[0,13]=18/90
    cargo[1,1]=1;cargo[1,11]=.4;cargo[1,12]=6/30;cargo[1,13]=12/30
    cargo[2,3]=1;cargo[2,11]=.2;cargo[2,12]=-1;cargo[2,13]=-1;cargo[2,14]=1
    limits=torch.tensor([t.limit for t in ORDER_TYPES]);f=urgency_features(cargo,limits)
    assert len(URGENCY_NAMES)==11
    assert f[2,0]==pytest.approx(2/3);assert f[2,1]==pytest.approx(.7/2)
    assert f[2,3]==pytest.approx(6/30);assert f[2,4]==pytest.approx(12/30)
    assert f[2,5]==pytest.approx(2/3);assert f[2,6]==pytest.approx(1/3);assert f[2,7]==pytest.approx(.2/2)
    assert f[0,8]==1 and f[0,9]==pytest.approx(6/30)
    empty=cargo.clone();empty[:,0]=0
    assert torch.equal(urgency_features(empty,limits),torch.zeros(6,11))


@pytest.mark.parametrize('architecture',['urgency','coordinated'])
def test_symmetry_padding_and_single_head_change(architecture):
    torch.set_num_threads(1);torch.manual_seed(10);net=build_network(architecture);base=build_network('destination')
    state={k:v for k,v in net.state_dict().items() if k in base.state_dict() and base.state_dict()[k].shape==v.shape}
    base.load_state_dict(state,strict=False);obs,mask=real_state();a,v=net(obs);b,w=base(obs)
    torch.testing.assert_close(v,w,atol=0,rtol=0)
    torch.testing.assert_close(a[:12] if architecture=='urgency' else a[12:],b[:12] if architecture=='urgency' else b[12:],atol=0,rtol=0)
    p=torch.randperm(12);c=torch.randperm(3)
    other=torch.cat((obs[:31],obs[31:355].reshape(12,27)[p].flatten(),obs[355:].reshape(3,16)[c].flatten()))
    changed,value=net(other);torch.testing.assert_close(changed,torch.cat((a[:12][p],a[12:])),atol=2e-6,rtol=1e-5);torch.testing.assert_close(value,v)
    empty=torch.zeros(403);noisy=torch.randn(403);noisy[:31]=0;noisy[31:355].reshape(12,27)[:,0]=0;noisy[355:].reshape(3,16)[:,0]=0
    for x,y in zip(net(empty),net(noisy)):torch.testing.assert_close(x,y,atol=0,rtol=0)
    for shape in [(403,),(4,403),(2,3,403)]:
        output=net(obs.expand(shape));assert output[0].shape==(*shape[:-1],26);assert all(torch.isfinite(t).all() for t in output)
    if architecture=='coordinated':
        _,_,h,pending,_=state_and_destinations(net,obs)
        dest=pending[:,11:17];counts=dest.sum(0)/12
        torch.testing.assert_close(h[:,97],counts) # cargo32 + pending64 + ncargo
        assert h.shape==(6,107)


@pytest.mark.parametrize('architecture',['destination','urgency','coordinated'])
def test_equivalent_inference_and_tie_guard(architecture):
    torch.manual_seed(20);model=build_network(architecture);obs,mask=real_state();fast=DestinationInferencePolicy(model)
    for x,m in [(obs,mask),(obs.repeat(8,1),mask.repeat(8,1))]:
        a,v=model(x);b,w=factored_forward(model,x);torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5);torch.testing.assert_close(v,w,atol=0,rtol=0)
        assert torch.equal(model.act(x,m,True)[0],fast.act(x,m,True)[0])
    # Force equal legal logits, proving reference fallback is actually exercised.
    for head in [model.pick_head[-1],model.deliver_head[-1],model.other_head]:
        with torch.no_grad():head.weight.zero_();head.bias.zero_()
    assert torch.equal(model.act(obs,mask,True)[0],fast.act(obs,mask,True)[0]);assert fast.fallback_states>0


@pytest.mark.parametrize('architecture',['urgency','coordinated'])
def test_real_collector_backward_reload(architecture,tmp_path):
    train(TrainConfig(architecture=architecture,total_steps=1024,eval_split='train',evaluate_initial=False,evaluate_baselines=False,learning_rate_schedule='late_linear_10pct'),tmp_path/architecture)
    state=load_checkpoint(tmp_path/architecture/'last.pt');net=build_network(state['architecture']);net.load_state_dict(state['model_state_dict'])
    assert all(torch.isfinite(p).all() for p in net.parameters())
    assert all(float(s['step'])>0 for s in state['optimizer_state_dict']['state'].values())


def test_type_bootstrap_masked_identity_and_shift():
    left=np.full((5,3,20),.6);left[:,0,:2]=np.nan;right=left-.1
    result=type_bootstrap(left,right,repetitions=2000)
    for load,r in result['loads'].items():
        assert r['estimate']==pytest.approx(.1);assert r['joint_95_percentile']==pytest.approx([.1,.1])
        assert r['eligible_days']==(18 if load=='low' else 20)
    bad=right.copy();bad[1,0,0]=.5
    with pytest.raises(ValueError):type_bootstrap(left,bad)
