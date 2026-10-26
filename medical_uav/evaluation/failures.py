"""Read-only failure classification. Counterfactual optimal actions are unknown."""
from collections import Counter
from medical_uav.evaluation.congestion import CongestionDiagnostics
from medical_uav.env.observation import rank_pending,direct_completion_time
from medical_uav.env.actions import decode
from medical_uav.env.config import EPS,ORDER_TYPES


class FailureDiagnostics(CongestionDiagnostics):
    def __init__(self):
        super().__init__();self.first_seen={};self.pick_state={};self.crossings={};self.opportunities=Counter();self.before_t=0

    def before(self,env,action):
        super().before(env,action);sim=env.sim;self.before_t=sim.t
        for o in rank_pending(sim):
            self.first_seen.setdefault(o.id,{'t':sim.t,'direct_slack':o.deadline-sim.t-direct_completion_time(sim,o)})
        kind,index=decode(action)
        if kind=='pick':
            order=rank_pending(sim)[index]
            self.pick_state[order.id]={'deadline_remaining':order.deadline-sim.t,'direct_slack':order.deadline-sim.t-direct_completion_time(sim,order),'onboard_boxes':len(sim.onboard)}
        for o in sim.onboard:
            if o.status=='onboard' and o.deadline-sim.t<=15:
                self.opportunities[(o.type_idx,'urgent_cargo_decisions')]+=1
                self.opportunities[(o.type_idx,'urgent_cargo_'+kind)]+=1
        self.pending_cross=[(o.id,o.type_idx,o.deadline,o.status) for o in sim.orders if o.status in ('pending','onboard') and o.deadline>=sim.t-EPS]
        self.action_kind=kind

    def after(self,env,action):
        super().after(env,action)
        lookup={o.id:o for o in env.sim.orders}
        for oid,typ,deadline,status in self.pending_cross:
            order=lookup[oid]
            # A successful delivery at/before deadline is not a missed crossing.
            if self.before_t<=deadline<env.sim.t-EPS and not (order.status=='delivered' and order.end_time<=deadline+EPS):
                self.crossings.setdefault(oid,{'kind':self.action_kind,'status_before':status})

    def finish(self,env):
        result=super().finish(env);sim=env.sim;failures=[]
        counts={t.name:Counter() for t in ORDER_TYPES}
        for o in sim.orders:
            if o.status=='future':continue
            c=counts[ORDER_TYPES[o.type_idx].name];c['released']+=1
            seen=self.first_seen.get(o.id);picked=self.pick_state.get(o.id)
            c['ever_visible']+=int(seen is not None)
            if o.status=='delivered' and o.end_time<=o.deadline+EPS:continue
            outcome='late' if o.status=='delivered' else 'expired_after_pick' if o.pickup_time is not None else 'expired_unpicked'
            c[outcome]+=1;c['failed_never_visible']+=int(seen is None)
            c['failed_first_visible_direct_late']+=int(seen is not None and seen['direct_slack']<0)
            c['failed_pick_already_direct_late']+=int(picked is not None and picked['direct_slack']<0)
            crossed=self.crossings.get(o.id)
            if crossed:c['deadline_cross_'+crossed['status_before']+'_'+crossed['kind']]+=1
            failures.append({'order_id':o.id,'type':ORDER_TYPES[o.type_idx].name,'outcome':outcome,'origin':o.origin,'destination':o.dest,'first_visible':seen,'pre_pick':picked,'deadline_crossing':crossed})
        result['failure_diagnosis']={'counts_by_type':{k:dict(v) for k,v in counts.items()},'urgent_cargo_decision_counts':{t.name:{k[1]:v for k,v in self.opportunities.items() if k[0]==i} for i,t in enumerate(ORDER_TYPES)},'failed_orders':failures,'scope':'observational; first-visible and direct timing ignore route alternatives/capacity/energy; crossings during an action do not prove that action was avoidable'}
        return result
