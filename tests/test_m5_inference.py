"""Inference-only algebra, mask and real-state decision equivalence."""
import pytest
import torch
from medical_uav.agents.networks import StructuredActorCritic
from medical_uav.agents.inference import FactoredStructuredInference,masked_argmax
from medical_uav.training.collector import Collector


def test_factored_inference_preserves_weights_logits_gradients_and_decisions():
    torch.manual_seed(19);torch.set_num_threads(1)
    reference=StructuredActorCritic();optimized=FactoredStructuredInference()
    optimized.load_state_dict(reference.state_dict())
    assert optimized.state_dict().keys()==reference.state_dict().keys()
    collector=Collector(8,19,('low','medium','high'));buffer,_=collector.collect(reference,32);collector.close()
    obs=buffer.obs.flatten(0,1);masks=buffer.action_mask.flatten(0,1)
    a,v=reference(obs);b,w=optimized(obs)
    torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5);assert torch.equal(v,w)
    assert torch.equal(reference.act(obs,masks,True)[0],masked_argmax(optimized,obs,masks))
    assert torch.equal(reference.act(obs[0],masks[0],True)[0],masked_argmax(optimized,obs[0],masks[0]))
    (a.sum()+v.sum()).backward();(b.sum()+w.sum()).backward()
    for p,q in zip(reference.parameters(),optimized.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=2e-4,rtol=1e-4)


def test_argmax_rejects_empty_masks_and_near_ties_use_reference():
    model=FactoredStructuredInference();obs=torch.zeros(403);mask=torch.ones(26,dtype=torch.bool)
    # Force all outputs to tie, exercising the historical tie path.
    with torch.no_grad():
        for p in model.parameters():p.zero_()
    assert masked_argmax(model,obs,mask)==model.act(obs,mask,True)[0]
    mask.zero_()
    with pytest.raises(ValueError):masked_argmax(model,obs,mask)
