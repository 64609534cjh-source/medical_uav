"""Read-only top-12 diagnostics, never exposed to neural policy inputs.

Hidden orders are simulator instrumentation, not extra observations. Matching a
visible/cargo destination describes a possible batching partner, not feasible
service or a causal truncation effect. Counts are decision-state weighted.
"""
from collections import Counter
from medical_uav.evaluation.diagnostics import EpisodeDiagnostics
from medical_uav.env.observation import rank_pending, direct_completion_time
from medical_uav.env.config import EPS


class CongestionDiagnostics(EpisodeDiagnostics):
    def __init__(self):
        super().__init__();self.congestion=Counter()

    def before(self,env,action):
        super().before(env,action)
        sim=env.sim;visible=rank_pending(sim);ids={o.id for o in visible}
        hidden=[o for o in sim.pending if o.id not in ids]
        cargo_dest={o.dest for o in sim.onboard}  # expired physical cargo retained
        visible_dest={o.dest for o in visible}
        matching=[o for o in hidden if o.dest in cargo_dest|visible_dest]
        direct=[o for o in matching if sim.t+direct_completion_time(sim,o)<=o.deadline+EPS]
        c=self.congestion
        c['states']+=1;c['over_12_states']+=int(len(sim.pending)>12)
        c['pending_sum']+=len(sim.pending);c['hidden_order_state_sum']+=len(hidden)
        c['hidden_matching_visible_or_cargo_states']+=bool(matching)
        c['hidden_matching_cargo_states']+=any(o.dest in cargo_dest for o in hidden)
        c['hidden_direct_on_time_matching_states']+=bool(direct)
        c['hidden_matching_order_state_sum']+=len(matching)
        c['max_pending']=max(c['max_pending'],len(sim.pending))

    def finish(self,env):
        result=super().finish(env)
        keys=['states','over_12_states','pending_sum','hidden_order_state_sum','hidden_matching_visible_or_cargo_states','hidden_matching_cargo_states','hidden_direct_on_time_matching_states','hidden_matching_order_state_sum','max_pending']
        result['congestion']={k:self.congestion[k] for k in keys}
        return result
