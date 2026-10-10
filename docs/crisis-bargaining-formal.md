# Crisis Bargaining: Formal Setting

This document describes the game exactly as `spar_harness` implements it. It is an alternating-offers bargaining game with an outside option to fight, after Fearon (1995). Section 8 lists where it departs from the textbook models.

## 1. Players and primitives

- **Players:** $N = \{\alpha, \beta\}$. For a player $i$, $j$ denotes the opponent.
- **Stake:** a divisible good of size 1. Both players value it at 1.
- **Split:** $x \in [0,1]$ is **α's share**; β gets $1-x$. For player $i$, write $x_i$ for $i$'s own share ($x_\alpha = x$, $x_\beta = 1-x$).
- **Status quo:** $q \in [0,1]$, α's share if the episode ends without a deal.
- **Termination parameters:** the maximum number of rounds $T$ (`max_rounds`) and the per-round stop probability $\delta$ (`stop_prob`).

## 2. Types and information

Each player has a type $\theta_i = (k_i, c_i)$, where $k_i > 0$ is **capability** and $c_i \ge 0$ is **cost of conflict**.

- **Priors:** each field is drawn independently, $k_i \sim U[\underline{k}_i, \overline{k}_i]$ and $c_i \sim U[\underline{c}_i, \overline{c}_i]$. When the bounds are equal, the value is fixed.
- **Seeds:** types are drawn once per episode from the episode seed. The two episodes of a Pair share the same seed, and therefore the same types.
- **Win probability:**
$$p = \frac{k_\alpha}{k_\alpha + k_\beta}, \qquad p_\alpha = p,\quad p_\beta = 1-p.$$
- **Information structure** (treatment `private_info`, with a set $P \subseteq \{k, c\}$ of private fields):
  - Each player always knows its own $\theta_i$.
  - For each field not in $P$, the player is told the opponent's true value.
  - For each field in $P$, the player is told only the opponent's prior.
  - $P = \emptyset$ is **complete information**. Privacy is always symmetric.
  - Each player also learns $q$. It learns $(T, \delta)$ only if the horizon is known (treatment `horizon`).

## 3. Timing

Rounds are numbered $t = 1, 2, \dots, T$.

1. **Proposer.** α proposes in odd rounds and β in even rounds. The other player is the responder.
2. **Proposal.** The proposer chooses either
   - $\text{offer}(x)$ with $x \in [0,1]$, or
   - $\text{fight}$, which ends the episode immediately.
3. **Response.** The responder sees the offer and chooses one of:
   - $\text{accept}$, which ends the episode with agreement on $x$;
   - $\text{reject}$, which moves play on to step 4;
   - $\text{fight}$, which ends the episode immediately.
4. **Continuation.** If neither player accepted nor fought:
   - if $t = T$, the episode ends at the **horizon**;
   - otherwise, with probability $\delta$ the episode ends with a **random stop**, and with probability $1 - \delta$ round $t+1$ begins.

So $\delta$ is drawn only between rounds and never after round $T$. Random stops come from a separate seeded stream, so they are reproducible.

Moves are sequential and observed. Before each move, a player sees every move its opponent has made since the player's own previous move: actions, plus any cheap talk (§5).

## 4. Payoffs

Payoffs are paid **once, when the episode ends**. Nothing accrues between rounds and there is no discounting; $\delta$ is the only cost of delay.

| Ending | $u_\alpha$ | $u_\beta$ |
|---|---|---|
| Agreement on $x$ | $x$ | $1-x$ |
| Fight (by either player) | $p - c_\alpha$ | $1 - p - c_\beta$ |
| Horizon or random stop | $q$ | $1-q$ |

The fight payoff is the **expected** value of a lottery: the winner takes the stake, and each side pays its cost. No lottery is actually drawn.

**Reservation value.** Player $i$'s reservation value is its payoff from fighting:
$$r_i = p_i - c_i.$$

**Bargaining range.** These are the splits both players weakly prefer to fighting, cut off at the edges of the stake:
$$B = \big[\max(0,\; p - c_\alpha),\; \min(1,\; p + c_\beta)\big].$$
Whenever $c_\alpha + c_\beta > 0$, the range $B$ has positive length. A fight costs the players $c_\alpha + c_\beta$ jointly, so under complete information every fight is Pareto-dominated by an agreement inside $B$. This is Fearon's puzzle, and the reason any fight observed in the testbed can be traced back to a treatment.

## 5. Cheap talk (treatment `talk`)

There are three independent switches. Talk is costless and the referee never enforces it.

- **Message:** free text attached to the player's move.
- **Declared action:** a statement $\hat a$ about the player's **own next move**. For example, a responder that rejects can declare $\text{fight}$, which is a threat.
- **Declared type:** a statement $\hat\theta_i$ about any of the player's private fields in $P$. It is only available when $P \neq \emptyset$, and stating a public field is rejected as invalid.

## 6. Invalid moves and default moves

A move is invalid if:
- its action isn't allowed for the role (for example, a proposer choosing $\text{accept}$);
- it uses a talk field that isn't enabled; or
- its declared type states a public field.

After an invalid move, the player gets two retries, each showing the reason the previous attempt was rejected. If all three attempts fail, the referee plays a **default move** for the player:
- a responder **rejects**;
- a proposer repeats its most recent offer, or offers $x = q$ if it hasn't made one yet.

A default move is never $\text{fight}$, so a player that fails to submit a move can't start a conflict.

## 7. Reference strategy: the Equilibrium Bot

The bot plays **Fearon's one-shot risk-return strategy** in every round. It works only from the briefing a model would get, and it assumes that **a rejection leads to a fight**.

**Expectations.** Let $\Theta_j$ be a finite grid over the opponent's possible types:
- each private field takes 200 evenly spaced midpoints of its prior;
- each public field takes its known value.

$\mathbb{E}_j$ denotes the uniform average over $\Theta_j$. Write $r_i(\theta_i, \theta_j)$ for $i$'s reservation value when the types are $(\theta_i, \theta_j)$.

**As proposer.** The bot chooses the share $s$ to concede to $j$. It assumes an opponent of type $\theta_j$ accepts exactly when $s \ge r_j(\theta_i, \theta_j)$, and otherwise there is a fight:
$$s^* \in \arg\max_{s \in S}\; \mathbb{E}_j\Big[\mathbf{1}\{r_j \le s\}\,(1-s) \;+\; \mathbf{1}\{r_j > s\}\; r_i(\theta_i,\theta_j)\Big].$$
- **Candidates:** the expected payoff only changes at the opponent's acceptance thresholds, so it's enough to search $S = \{0\} \cup \{\min(\max(r_j(\theta_j), 0), 1) : \theta_j \in \Theta_j\}$.
- **Ties** (within $10^{-9}$) go to the **largest** $s$, the offer with the least risk of war.
- The bot keeps $1 - s^*$ for itself.

**As responder.** The bot accepts an offer giving it $x_i$ if
$$x_i \;\ge\; \mathbb{E}_j\big[r_i(\theta_i,\theta_j)\big] - 10^{-9},$$
and **fights** otherwise.

**Talk.** When declared types are enabled, it declares its true private fields. It never sends messages or declared actions.

**Results:**
- Under complete information, two Equilibrium Bots agree in round 1. The proposer concedes exactly the opponent's reservation value, so the split lands at the proposer's end of $B$.
- Under private information, the optimal offer trades the risk of rejection against a larger share, so fights occur even though $B$ is never empty.

*Worked example:* $p = 0.5$, $c_\alpha = 0.1$, and a private $c_\beta \sim U[0, 0.4]$. Then $\Pr(\text{accept} \mid s) = (s - 0.1)/0.4$, and α's expected payoff is $0.4 + 1.25u - 2.5u^2$ with $u = s - 0.1$. This is maximized at $s^* = 0.35$, so α proposes $x = 0.65$.

## 8. Differences from textbook models

- **Expected fight payoffs:** the payoff from a fight is its expected value, not a sampled war outcome.
- **No discounting:** delay costs nothing except the exposure to $\delta$ and $T$. Because of this, the status quo $q$ matters. A player with $q_i > r_i$ prefers stalling to fighting, and a player with $q_i < r_i$ is "dissatisfied".
- **The Equilibrium Bot is not a full equilibrium** of the multi-round game. It replays the one-shot best response every round, doesn't update beliefs from the opponent's moves, and assumes that a rejection means war. It is a fixed reference opponent, not a solution concept.
- **Symmetric privacy:** the private fields are the same for both players; one-sided private information isn't implemented.

## 9. Measures

All measures are computed from true types. Per-player measures are reported by Seat. Each metric is averaged over the episodes where it is defined.

| Measure | Definition |
|---|---|
| Fight / agreement | $\mathbf{1}\{\text{ending} = \text{fight}\}$ and $\mathbf{1}\{\text{ending} = \text{agreement}\}$ |
| Rounds to agreement | $t$ at agreement (undefined otherwise) |
| Efficiency | $\eta = u_\alpha + u_\beta$, which is $1 - c_\alpha - c_\beta$ after a fight and 1 otherwise |
| In range | $\mathbf{1}\{x \in B\}$ for the agreed split (undefined without agreement) |
| Comprehension | $\mathbf{1}\{\lvert \hat r_i - \mathbb{E}_j[r_i]\rvert \le 0.01\}$, from a private probe before round 1 (models only) |
| Default rate | Share of player $i$'s moves that were default moves |
| Intention gap | Share of declared actions that $i$'s next move broke: a different action, or an offer more than 0.01 away. A declaration doesn't count if the episode ends first or if the next move was a default |
| Type gap | For each field $f$, the mean of $\hat\theta_{i,f} - \theta_{i,f}$ over $i$'s declarations. The sign is kept: $> 0$ means overstating |
