**Purpose and setup.** Before PPO, our main method, we checked that both networks can learn by imitating a teacher (behaviour cloning): mainly Slack-aware, and for comparison the rollout planner, stronger but far slower. Each teacher supplied 720 training and 180 validation days (demand seeds 0–299 at each of three demand levels, or loads, split by seed); Table 1 uses separate days. Training minimises the cross-entropy to the teacher's action (Adam, learning rate 3·10⁻⁴, batch 256, up to 100 epochs, lowest-validation-loss epoch kept); the value head is left for PPO.

{{table1}}

**Result.** Imitating Slack-aware, the set network agrees with the teacher on 97.4% of validation decisions and matches its daily on-time rate; the MLP agrees on 94.2% and delivers 0.8 and 2.2 percentage points fewer orders on time at medium and high load (Table 1, Figure 2). Imitating the rollout planner, agreement is 74.6% (MLP) and 78.0% (set network), yet at these loads both deliver more orders on time than when imitating Slack-aware. With either teacher the set network beats the MLP at high load, by 2.4 and 4.9 points (paired 95% intervals above zero), with a sixth of the parameters. Served rates move the same way, so no network trades expired orders for on-time deliveries.

{{fig2}}
