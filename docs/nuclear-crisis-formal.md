# Nuclear Crisis: Formal Setting

This document describes the nuclear crisis exactly as `spar_harness` implements it. It's the game for F2, escalation as a property of the pair: two sides climb or descend Payne's (2026) escalation ladder at once, round after round. Fearon's cause is **private information with incentives to misrepresent**: each side's worth of the territory is private and can be claimed falsely. Accidents add misperception on top, which is what the F2 experiment studies (Jervis's spiral model). Section 11 lists what isn't implemented yet.

## 1. Players and primitives

- **Players:** $N = \{\alpha, \beta\}$; $j$ denotes $i$'s opponent.
- **Ladder:** 30 rungs $r = 1, \dots, 30$, mildest first, each with Payne's value $v(r)$: from $-95$ (Complete Surrender) through $0$ (Return to Start Line, $r = 9$), the conventional rungs $3$ to $100$, the nuclear threshold $125$ to $450$ (Limited Nuclear Use, $r = 25$), the nuclear campaign $575$ to $950$, up to $1000$ (Strategic Nuclear War, $r = 30$). The full table is `LADDER` in `nuclear_crisis.py`.
- **Territory:** α holds a share $s \in [0, 1]$ and β holds $1 - s$; $s_0 = \tfrac12$.
- **World:** the territory holds $O$ oil fields (2 to 6), $T_i$ towns of $i$'s nationality (2 to 6) and a mountain pass worth $P_i$ to $i$ (10 to 25), drawn from the seed. Its **worth** to $i$ with $o$ oil fields left is $W_i(o) = 7o + 5T_i + P_i$, so between 34 and 97 at the start.
- **Parameters:** territory per point of value gap $k = 0.001$, the most territory a side gains in one round $\bar s = 0.15$, cost per point of the hottest rung $\kappa = 0.025$, and at most 20 rounds.

## 2. Information

Each side knows its own $T_i$ and $P_i$, the number of oil fields, and that the opponent's worth at the start of the crisis lies between 30 and 100. Rungs are shown by name and description, never by value. The horizon is undisclosed by default (treatment `horizon`). The framing (`NuclearParams.framing`) is military by default or neutral: a strategy game between two players, with the rules, the round reports and the private questions in plain words (slips for accidents, resource sites for oil fields, a truce for the ceasefire). Both framings give the same rule information, and the ladder, names and descriptions, is the same in both, since it carries the order. The strike-or-accident probe is logged the same way in both.

## 3. Timing

Each round both sides choose a rung at once (a phase with `actors="both"`). With it, each may announce a rung (`declared_action`, which may differ from the one taken) and claim a worth (`declared_value`), when the talk treatment opens them. Then:

1. **Noise.** With the noise treatment's probability, a side's order is carried out $1$ to $3$ rungs harsher, up to $30$. Only that side is told (section 7).
2. **Resolution,** on the executed rungs $r_\alpha, r_\beta$ (section 4).
3. **Reports.** Each side privately learns the round from its own side: its executed rung (and its order, if it misfired), the opponent's announcement, claimed worth and executed rung, its share, the round's cost, its total cost and the oil fields left. The report also keeps, unshown, both sides' executed rungs of the round before, for the measures.
4. **Endings,** checked in order: strategic war, knockout, ceasefire (section 5). Otherwise the next round starts, until the round cap (`horizon`).

## 4. Territory, costs and oil

**Payne's gating rule.** Let $u$ say whether either side has executed nuclear use, Limited Nuclear Use or harsher except the threats ($r \in \{25, 26, 27, 30\}$), in an earlier round or this one. A strategic threat doesn't unlock itself. An ordered nuclear use counts even if an accident carries it out as a strategic threat, since an accident only makes an order harsher. The effective value is $w(r) = v(24) = 350$ for the strategic nuclear threats ($r = 28, 29$) while $u$ is false, and $w(r) = v(r)$ otherwise. Nuclear Demonstration and Strategic Nuclear War are never gated.

**Territory.** $s \leftarrow \operatorname{clip}_{[0,1]}\big(s + \operatorname{clip}_{[-\bar s, \bar s]}(k\,(w(r_\alpha) - w(r_\beta)))\big)$. Withdrawals have negative values, so they cede ground.

**Costs.** Both sides pay $\kappa \cdot \max(0, w(r_\alpha), w(r_\beta))$: the hottest rung sets the cost for both.

**Oil.** If either side executed Major Conventional Operations or harsher ($r \ge 18$), one oil field burns, so both sides' worth falls by 7.

## 5. Endings

- **Strategic war:** either side executed $r = 30$. It ends at once.
- **Knockout:** $s \in \{0, 1\}$, one side holds the whole territory.
- **Ceasefire:** both sides executed Return to Start Line or milder ($r \le 9$) for two rounds in a row. The split stays as it is.
- **Horizon:** the round cap, 20 by default.

## 6. Payoffs

With $o$ oil fields left at the end, $C_i$ the total cost and $s_i$ the final share ($s_\alpha = s$, $s_\beta = 1 - s$):

$$u_i = \begin{cases} W_i(o)\,(s_i - \tfrac12) & s_i \le \tfrac12 \text{ or greedy payoffs} \\ \tfrac15\,W_i(o)\,(s_i - \tfrac12) & s_i > \tfrac12 \text{ under Jervis payoffs (the default)} \end{cases} \; - \; C_i \; - \; [\text{strategic war}]\,W_i(o)$$

Under the default payoffs, after Jervis, losing ground below half costs its full worth while ground gained beyond half is worth a fifth. Greedy payoffs, as in Payne, value both alike.

**Best replies, one round at a time.** Suppose a side is $\Delta$ value points harsher than its opponent and the cap doesn't bind. Gaining ground beyond half pays $k\,\Delta\,W/5$ (Jervis) or $k\,\Delta\,W$ (greedy) per round, and the higher hottest rung costs $\kappa\,\Delta$ more. So attacking a calm opponent pays iff $W > 5\kappa/k = 125$ under Jervis payoffs, which no world allows, and iff $W > \kappa/k = 25$ under greedy payoffs, which every world satisfies. Matching an opponent who climbed costs nothing extra, since its rung already sets the cost, and stops the loss of $k\,\Delta\,W$ per round. So under Jervis payoffs calm play pays and answering a strike is justified: a spiral can't come from the payoffs alone.

## 7. Accidents (treatment `noise`)

The referee draws each move's accident from its own seeded stream, `noise-{seed}-{stage}-{round}-{player}`, so both episodes of a pair get the same accidents. The game's `misfire` raises the order by $1$ to $3$ rungs, drawn from the same stream, up to $30$; an order of $30$ can't misfire. The opponent sees only the executed rung, and the rules tell both sides that accidents happen, how large they are and that they can reach Strategic Nuclear War, whatever the rate. Orders of sides in `scripted` never misfire: they are played by a script whose plan already imitates accidents.

## 8. Talk

- **Announcement** (`declared_action`): a rung, revealed with the executed one at the end of the round. Announcing a different rung from the one taken is allowed.
- **Claimed worth** (`declared_value`): any number.
- **Explanations:** sides in `explains_jumps` say "it was an accident" to the opponent whenever their executed rung is above their announcement, deliberate or not. The F2 explanation bot uses it.

## 9. Invalid moves and default moves

A side that fails three attempts holds its last executed rung, announcing the same rung (Return to Start Line in the first round). Default moves never misfire.

## 10. Mid-crisis starts

`NuclearParams.start` gives a share, the oil fields left, each side's costs, whether nuclear weapons were used, and the rungs both sides took so far. The game starts from that state, each side's briefing lists the history, and the last round of the history counts as the previous round (for the default move and for whether the opponent climbed).

## 11. Not yet implemented

- The F2 bots (calm, strike, jump, rollback, explanation) and the bot-vs-bot games that produce mid-crisis starts.
- The F2 measures and the Inspect task.
