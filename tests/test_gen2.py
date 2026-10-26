"""Observable grouping, set symmetries, padding, and real PPO integration."""
import numpy as np
import pytest
import torch
from medical_uav.agents.networks import build_network, masked_distribution, StructuredActorCritic
from medical_uav.agents.relational import destination_groups, PENDING_DEST, CARGO_DEST, TRAVEL_START
from medical_uav.training.trainer import TrainConfig, train, load_checkpoint
from tests.test_structured import real_state


@pytest.mark.parametrize('architecture', ['destination', 'relational'])
def test_set_symmetry_and_empty_padding(architecture):
    torch.set_num_threads(1);torch.manual_seed(12)
    net=build_network(architecture)
    obs,mask=real_state();p=torch.randperm(12);c=torch.randperm(3)
    changed=torch.cat((obs[:31],obs[31:355].reshape(12,27)[p].flatten(),obs[355:].reshape(3,16)[c].flatten()))
    a,v=net(obs);b,w=net(changed)
    torch.testing.assert_close(b,torch.cat((a[:12][p],a[12:])),atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(v,w,atol=2e-6,rtol=1e-5)
    prob=masked_distribution(a,mask).probs
    torch.testing.assert_close(masked_distribution(b,torch.cat((mask[:12][p],mask[12:]))).probs,torch.cat((prob[:12][p],prob[12:])))
    empty=torch.zeros(403);noisy=torch.randn(403);noisy[:31]=0
    noisy[31:355].reshape(12,27)[:,0]=0;noisy[355:].reshape(3,16)[:,0]=0
    for x,y in zip(net(empty),net(noisy)):torch.testing.assert_close(x,y,atol=0,rtol=0)
    net.train();x=net(obs);net.eval();y=net(obs)
    for a,b in zip(x,y):torch.testing.assert_close(a,b,atol=0,rtol=0)
    for shape in [(403,),(2,403),(2,3,403)]:
        outputs=net(obs.expand(shape));assert outputs[0].shape==(*shape[:-1],26)
        assert all(torch.isfinite(t).all() for t in outputs)
    # A real unpickable order remains a key/state input; only presence pads it.
    changed=obs.clone();rows=changed[31:355].reshape(12,27);i=int(torch.where(rows[:,0]>0)[0][0]);rows[i,-1]=0
    before=net(changed);rows[i,18]+=1;after=net(changed)
    assert not torch.equal(before[1],after[1])


def test_destination_groups_include_physical_expired_cargo_and_only_change_deliver_head():
    torch.manual_seed(2);net=build_network('destination');base=StructuredActorCritic()
    shared={k:v for k,v in net.state_dict().items() if k in base.state_dict() and k.split('.')[0]!='other_head'}
    base.load_state_dict(shared,strict=False)
    with torch.no_grad():base.other_head.weight[6:]=net.other_head.weight;base.other_head.bias[6:]=net.other_head.bias
    obs,_=real_state();a,v=base(obs);b,w=net(obs)
    torch.testing.assert_close(a[:12],b[:12],atol=0,rtol=0);torch.testing.assert_close(a[18:],b[18:],atol=0,rtol=0);torch.testing.assert_close(v,w,atol=0,rtol=0)
    cargo=torch.zeros(3,16);cargo[0,0]=1;cargo[0,CARGO_DEST+2]=1;cargo[0,14]=1
    emb=torch.tensor([[1.,2.],[8.,9.],[3.,4.]])
    pool,count=destination_groups(cargo,emb,CARGO_DEST)
    assert count[2]==1;assert count.sum()==1;torch.testing.assert_close(pool[2],emb[0]);assert (pool[[0,1,3,4,5]]==0).all()
    assert PENDING_DEST==11 and CARGO_DEST==5 and TRAVEL_START==16


@pytest.mark.parametrize('architecture',['destination','relational'])
def test_actual_collector_ppo_backward_reload(architecture,tmp_path):
    config=TrainConfig(architecture=architecture,total_steps=1024,learning_rate_schedule='late_linear_10pct',evaluate_initial=False,evaluate_baselines=False,eval_split='train')
    train(config,tmp_path/architecture)
    state=load_checkpoint(tmp_path/architecture/'last.pt');net=build_network(state['architecture']);net.load_state_dict(state['model_state_dict'])
    assert all(torch.isfinite(p).all() for p in net.parameters())
    assert all(float(v['step'])>0 for v in state['optimizer_state_dict']['state'].values())
    obs,mask=real_state();a,logp,v=net.act(obs,mask,True);new,ent,w=net.evaluate_actions(obs,mask,a)
    torch.testing.assert_close(logp,new);assert mask[a];assert torch.isfinite(ent)
