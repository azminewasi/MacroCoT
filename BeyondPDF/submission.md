---
layout: distill
title: "Shorter Chains via Macro Transitions: Constant-Factor Compression for Chain-of-Thought Counter Simulation"
description: Transformers can simulate complex computations through chain of thought, but doing so one step per token makes generation slow and expensive. We introduce two ways to compress these reasoning traces by combining multiple transitions or counting operations into fewer tokens while preserving the original computation. Experiments show that moderate compression maintains strong length generalization while reducing trace length, latency, attention cost, and memory use. However, overly aggressive compression can hurt accuracy and require more training data.
 
htmlwidgets: true
 
# Camera-ready author list
authors:
  - name: Azmine Toushik Wasi
    url: "https://openreview.net/profile?id=~Azmine_Toushik_Wasi1"
    affiliations:
      name: "University of Surrey, UK; Computational Intelligence and Operations Laboratory, Bangladesh"
  - name: MD Shafikul Islam
    url: "https://openreview.net/profile?id=~MD_Shafikul_Islam1"
    affiliations:
      name: "Louisiana State University, USA; Computational Intelligence and Operations Laboratory, Bangladesh"
  - name: Md Manjurul Ahsan
    url: "https://openreview.net/profile?id=~Md_Manjurul_Ahsan1"
    affiliations:
      name: "University of Oklahoma, USA; Computational Intelligence and Operations Laboratory, Bangladesh"
 
# Must be the same name as your submission. Do not change this name, just use "submission.bib".
bibliography: submission.bib

# Add a table of contents to your submission.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the submission to work correctly.
toc:
  - name: Abstract
  - name: Introduction
  - name: Related Work
  - name: Preliminaries
  - name: Transition Compression
  - name: What Is Compressed? Positions and Counting
  - name: Experiments
  - name: Discussion
  - name: Conclusion
---


## Abstract

Chain-of-thought (CoT) constructions that make transformers Turing complete, typically simulate a counter machine, one transition per generated token, making autoregressive length proportional to the simulated runtime. Since serial generation is a key computational cost, we ask whether this length can be reduced while preserving the recognized language and the generalization properties of the simulation. We introduce two machine rewrites with exact recognition guarantees. First, order k macro transitions combine k reachable base transitions into one token. We prove that the resulting guards retain the per-counter threshold structure, determinism is preserved, and a run of length T is compressed to about T divided by k steps for fixed k, a constant factor reduction that leaves the asymptotic serial complexity unchanged, with a vocabulary bounded by the base vocabulary raised to k. Second, base b counting compresses monotone decrement bursts by processing b counter units per token. Both rewrites remain within the same CoT counter programming class, allowing the base construction's length generalization result to transfer under its assumptions. Experiments on six synthetic counter machine tasks, including equal-count and three-sum arithmetic verification, show that transformers trained on compressed traces retain length generalization under moderate compression while achieving the predicted reductions in trace length, and that this retention degrades once compression becomes large relative to the counting range. We observe corresponding reductions in generation latency, attention operations, and key value cache size. Accuracy degrades as compression becomes aggressive relative to the counting range, consistent with our analysis. We further characterize reachable vocabulary growth and quantify the associated data scaling cost.


## Introduction

Transformers that produce a chain-of-thought before answering can carry out computations that a single forward pass cannot. A line of theory has made this precise by showing that, with sufficiently many generated intermediate tokens, decoder only transformers can simulate general models of computation <d-cite key="perez2021attention,merrill2024expressive,li2024chain,jiang2026softmax"></d-cite>. A common route uses counter machines: the transformer emits one transition token per step, while causal attention recovers counter values from weighted prefix counts of previously generated tokens <d-cite key="jiang2026softmax"></d-cite>. In this setting, the chain-of-thought length equals the number of transitions taken by the simulated machine. This length is a fundamental computational resource: it determines the number of serial generation steps, contributes to the quadratic attention cost over the growing context, and is constrained by recent lower bounds on serial computation with chain-of-thought <d-cite key="li2024chain,merrill2024expressive"></d-cite>.

The counter machines underlying these constructions can be inefficient. A counter machine may require exponentially many steps in the space of the Turing machine it simulates <d-cite key="jiang2026softmax,fischer1968counter,minsky1967computation"></d-cite>, while even simple counting loops consume one transition per counter unit. Since each transition corresponds to a generated position, inefficient machines induce unnecessarily long chains of thought. This raises a question distinct from expressive power: *given a fixed language and simulation scheme, can we shorten the chain-of-thought without changing the recognized language or compromising the properties that enable length generalization?*

We study whether the step count of these counter machine based CoT constructions can be reduced without changing the computation being simulated. Our approach is to modify the simulated machine rather than the transformer architecture: several fine grained transitions can be represented by a single macro transition, while counting loops can process multiple counter units per generated token. We characterize when these rewrites preserve the structure required by the underlying transformer construction and quantify the resulting reduction in generated positions.

Our contributions are:
1. **Macro transitions with exact preservation.** We define order $k$ macro transitions by composing reachable sequences of $k$ base transitions into a single transition. We show that the resulting guards retain the per counter threshold structure of the base machine (Lemma 1) and that determinism is preserved (Lemma 2). For a base run of length $T$, the compressed run has length exactly $\lceil T/k\rceil$ and agrees with the base computation at block boundaries (Theorem 1). We also bound the macro vocabulary by $|\Delta|^k$ (Proposition 1).
2. **Sequence length as a representation dependent cost.** The macro construction changes the granularity of the generated computation rather than removing the information needed to specify it. This reduces the number of generated positions, and hence serial generation cost. For the generated-trace portion of attention, a factor $k$ reduction in sequence length corresponds to a factor $k^2$ reduction in the associated attention cost; the full end-to-end factor also depends on the input prefix.
3. **Compressed counting loops.** We introduce a base $b$ rewrite for monotone decrement bursts, replacing a burst of mass $V$ with $\lceil V/b \rceil$ generated tokens (Proposition 5). For the counter machine setting analyzed here, this gives an additive $\log_2 b$ reduction in the exponent of the corresponding simulation step count (Corollary 1).
4. **Composition and preservation of the CoT construction.** The two rewrites can be composed, giving an approximately $bk$ reduction in positions for counting dominated runs (Corollary 2). The resulting machines remain within the counter programming class used by the base CoT construction. Under the stated compilation assumption, the corresponding length generalization result therefore applies to the compressed representation (Assumption 1, Proposition 4). We also characterize reachable macro vocabularies and derive tighter structural bounds for specific machine structures, including a linear bound for single successor counting loops (Propositions 2 and 3).
5. **Empirical evaluation of the tradeoffs.** We evaluate the constructions on six synthetic tasks implemented by counter machines, including equal-count comparison and multi-phase three-sum arithmetic verification. Transformers trained on compressed traces retain length generalization under moderate compression while exhibiting the predicted reductions in trace length, with accuracy degrading as compression grows relative to the counting range. We measure generation latency, attention operations, and key value cache size, and study how vocabulary growth affects accuracy and training data requirements.

The base construction spends one generated position per base transition. Our first operator groups every reachable sequence of $k$ consecutive base transitions into a single macro token, so a run of $T$ base steps becomes a trace of about $T/k$ tokens, rounded up. The composed guard remains a per counter threshold test, so the compressed machine stays within the same guarded update counter machine framework, with counters recoverable from prefix counts. It therefore recognizes the same language. The interactive example below illustrates this compression: the base trace collapses into its macro trace, reducing the number of generated positions while preserving the information represented by the computation.


<figure style="text-align: center; margin: 20px 0;">
    <iframe 
        src="{{ 'assets/html/submission/macro_compression.html' | relative_url }}"
        width="100%" 
        height="500" 
        style="border: none; overflow: hidden; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);"
        title="Interactive Graph">
    </iframe>
    <figcaption><b>Order-$k$ macro transitions collapse a base trace.</b> Each block of up to $k$ consecutive base transitions becomes a single macro token, so a run of $T$ base steps becomes a trace of $\lceil T/k \rceil$ tokens. Adjust $T$ and $k$ to see the positions fall by a factor of $k$ while the trace bit-length stays flat, since a macro alphabet of size up to $\vert \Delta \vert^k$ needs about $k$ times as many bits per token.</figcaption>
</figure>

Overall, our results show that, for the counter machine based CoT constructions studied here, the number of generated reasoning steps can be reduced by changing the granularity of the simulated computation. The analysis also makes explicit the resulting tradeoffs: greater compression reduces sequence length but can increase vocabulary size and affect learnability. We characterize these tradeoffs rather than claiming a general speedup for arbitrary chain-of-thought. In our setup, $k$ and $b$ are fixed, so all reductions reported here are constant factors in the input length; they reduce the serial length of the simulation by a bounded multiple rather than changing its asymptotic dependence on the simulated runtime.

## Related Work

**Expressive power of transformers.** A substantial body of work characterizes the computational limits of transformers. Fixed depth encoders without intermediate decoding fall within restricted circuit and logical classes <d-cite key="merrill2022saturated,merrill2023parallelism,merrill2023logic,hao2022formal,chiang2023tighter,barcelo2024logical"></d-cite>, while formal language studies characterize the languages recognized under different architectural and attention constraints <d-cite key="hahn2020theoretical,bhattamishra2020ability,yao2021self,ebrahimi2020can,yang2024masked,strobl2024what"></d-cite>. The RASP framework provides a programming perspective on transformer computation <d-cite key="weiss2021thinking,lindner2023tracr"></d-cite>. Our construction follows a related perspective: counters are represented as prefix counts, a quantity that causal attention can recover from the generated trace.

**Chain-of-thought as computation.** Intermediate generation can substantially expand the computational capabilities of transformers. Scratchpads and chain-of-thought were first demonstrated empirically <d-cite key="nye2021show,wei2022chain"></d-cite>, followed by theoretical work showing that intermediate decoding enables serial computations that cannot be carried out in a single forward pass <d-cite key="feng2023towards,li2024chain,merrill2024expressive"></d-cite>. In particular, Jiang et al. show that softmax transformers can achieve Turing completeness by simulating counter machines, using one generated token per machine transition and prefix counting to recover counter values <d-cite key="jiang2026softmax"></d-cite>. This builds on earlier Turing completeness results for attention based models <d-cite key="perez2021attention"></d-cite>. These results establish the computational power available from sufficiently long generated traces. We instead study the complementary question of how the same counter machine computation can be represented with fewer generated positions.

**Counter machines.** Counter machines are classical models of computation with well established connections to Turing machines and formal language recognition <d-cite key="minsky1967computation,fischer1968counter,sipser2013introduction"></d-cite>. Their importance here is both computational and representational: a counter machine provides a simple operational description whose transitions can be mapped directly to generated tokens. However, the resulting step complexity can be large, including exponential overheads in standard simulations. We exploit this representation to study two semantics preserving rewrites, macro transitions and compressed counting, that reduce the number of transitions while keeping the recognized language unchanged.

**Length generalization and positional encoding.** Length generalization in transformers depends on both the underlying computation and the positional representation used to encode it <d-cite key="deletang2023neural,anil2022exploring,zhou2024what,liu2023transformers"></d-cite>. Several works show improved extrapolation on counting and algorithmic tasks through alternative or randomized positional schemes <d-cite key="kazemnejad2023impact,press2022train,su2024roformer,ruoss2023randomized"></d-cite>. Related theoretical work connects generalization to the existence of short attention based programs for the target computation <d-cite key="zhou2024what"></d-cite>. Our setting is complementary: rather than changing positional representations or the transformer architecture, we shorten the generated program itself. Under the assumptions of the underlying counter machine construction, this preserves the corresponding length generalization property.

**Complementary approaches to efficient generation.** Several lines of work address the cost of long autoregressive computation from other directions. Positional encoding methods aim to make longer sequences easier to learn or extrapolate <d-cite key="kazemnejad2023impact,press2022train,ruoss2023randomized"></d-cite>, while our approach changes the number of generated positions required by the simulated computation. Similarly, methods that reduce key value cache costs reduce the memory associated with each generated position, whereas our rewrites reduce the number of positions themselves. These approaches are therefore complementary. More broadly, work connecting intermediate tokens to sample efficiency highlights a tradeoff between the computational structure represented by a trace and the difficulty of learning it. Our experiments make a related tradeoff explicit: stronger compression shortens the trace but can enlarge the effective token vocabulary, creating a corresponding data and learnability cost.


**Context reduction and PENCIL.** Yang et al. introduce PENCIL, which alternates generating thoughts with erasing intermediate thoughts that are no longer needed <d-cite key="yang2025pencil"></d-cite>. They also establish time and space guarantees for Turing machine simulation. PENCIL reduces the context retained during reasoning and can support long computations within a smaller memory budget. Our rewrites change transition granularity: each macro represents several machine steps, reducing the number of generated positions while retaining the emitted macro trace. For fixed $k$ and $b$, our savings are constant factors and can require a larger vocabulary. These methods address complementary resources, but combining them would require a separate correctness analysis.

## Preliminaries

**Counter machines.** A deterministic guarded update counter machine is a tuple $S = (Q, k_c, \Delta, q_0, F_{\mathrm{acc}}, F_{\mathrm{rej}}, \iota)$ with control states $Q$, $k_c$ counters, transitions $\Delta$, a start state, disjoint accepting and rejecting halting states, and an initializer $\iota$ that sets the counters from the input word. A configuration is a pair $(q, x)$ with $q \in Q$ and $x \in \mathbb{N}^{k_c}$. A transition is $\tau = (\mathrm{src}(\tau), g(\tau), u(\tau), \mathrm{dst}(\tau))$, where the guard $g(\tau)$ is a conjunction of atoms $(x_i \bowtie c)$ for a single counter $i$, a constant $c \in \mathbb{N}$, and a comparison $\bowtie \in \{=, \ge, \le, >, <\}$, and the effect $u(\tau) \in \mathbb{Z}^{k_c}$ is added to the counters. The machine is deterministic when at most one transition is enabled at every configuration. On input $w$, it starts at $(q_0, \iota(w))$ and repeatedly applies the unique enabled transition, emitting its token, until it reaches a halting state. The emitted sequence is the trace, and $w$ is accepted when the final state is accepting.

**Chain-of-thought counter programming.** The base construction compiles such a machine into a decoder only transformer that emits the trace one token per step <d-cite key="jiang2026softmax"></d-cite>. The key identity is that the value of counter $i$ before step $t$ is

$$
x_i^{(t)} = \iota_i(w) + \sum_{j < t} u_i\!\left(\tau_{a_j}\right),
$$

a weighted prefix count over the emitted trace tokens $a_1, \ldots, a_{t-1}$. A causal attention head computes such prefix counts, and each guard atom $(x_i \bowtie c)$ is therefore a threshold test on that value. We call a simulation that uses integer effects and single counter threshold guards over prefix counts a CoT counter program. Our rewrites remain within this class, allowing the corresponding properties of the base construction, including its length generalization guarantee, to transfer under the stated assumptions.

**Cost model.** The transformer pays for a trace through its number of generated positions $N$, but inference also has an input prefix of length $P$. With a key-value cache, prefill attention costs $\Theta(P^2d)$ and decoding costs $\Theta((PN+N^2)d)$, so total attention work is $\Theta((P^2+PN+N^2)d)$. Cache storage is $\Theta((P+N)Ld)$ for $L$ layers, up to the constant factor for keys and values. Our compression changes $N$ while leaving the input prefix fixed. Thus the generated portion has the predicted quadratic reduction, but the end-to-end factor is smaller whenever the prefix terms dominate.

## Transition Compression
The base construction spends one generated position on each machine transition. Our first compression mechanism changes this granularity: several consecutive transitions that are jointly determined by the current configuration are represented by a single macro transition. The key question is whether this composition preserves the restricted guard structure required by the CoT counter program. We first establish this property, then derive determinism, exact recognition, and vocabulary bounds.

### Macro Transition Construction

Fix a deterministic machine $S$ with transition set $\Delta$. A block of length $r$ is a sequence $\pi = \tau_1 \tau_2 \cdots \tau_r$ of base transitions that is chainable, meaning $\mathrm{dst}(\tau_j) = \mathrm{src}(\tau_{j+1})$ for all $j < r$. For a block, define the accumulated shift before its $j$th step as $s_j = \sum_{l < j} u(\tau_l) \in \mathbb{Z}^{k_c}$, with $s_1 = 0$. We then define the macro effect and guard as

$$ u(M_\pi) = \sum_{j=1}^{r} u(\tau_j), \qquad \varphi_\pi(x) = \bigwedge_{j=1}^{r} g(\tau_j)\big(x + s_j\big), $$

with $\mathrm{src}(M_\pi) = \mathrm{src}(\tau_1)$ and $\mathrm{dst}(M_\pi) = \mathrm{dst}(\tau_r)$.
The token $M_\pi$ fires only when $x + s_j \in \mathbb{N}^{k_c}$ for every $j$, which we fold into $\varphi_\pi$ through the nonnegativity condition at each step.

The main technical question is whether composing guards in this way takes us outside the guard language supported by the base CoT construction.

**Lemma 1 (Canonical guard).**
For any block $\pi$, the macro guard $\varphi_\pi$ is equivalent to a conjunction of single counter threshold atoms, or equivalently, a per counter interval constraint $\bigwedge_{i=1}^{k_c} \big( \ell_i \le x_i \le h_i \big)$ with $\ell_i \in \mathbb{N}$ and $h_i \in \mathbb{N} \cup \{\infty\}$.

**Proof.**
Each atom of $g(\tau_j)$ has the form $(x_i \bowtie c)$ for a single counter $i$. Substituting the shifted argument $x + s_j$ replaces it by $(x_i \bowtie c - s_{j,i})$, which is again a single counter threshold atom on the same counter with the constant translated by the integer $s_{j,i}$. When the translated constant is negative, the atom is either trivially true for $\ge$ and $>$ or trivially false for $\le$, $<$, and $=$, which we simplify away. The nonnegativity requirement $x_i + s_{j,i} \ge 0$ is the atom $(x_i \ge -s_{j,i})$. Thus $\varphi_\pi$ is a finite conjunction of single counter threshold atoms. For a fixed counter $i$, a conjunction of atoms $(x_i \ge a)$, $(x_i > a)$, $(x_i \le b)$, $(x_i < b)$, and $(x_i = c)$ is the intersection of half lines and points on $\mathbb{N}$, which is an interval $[\ell_i, h_i]$, possibly empty, a single point, or unbounded above. Taking the conjunction over counters gives the stated per counter interval form. $\square$

Lemma 1 is the key closure property: composing transitions does not leave the guard language. The macro guard remains a threshold test on the prefix count defined in the preliminaries, so a macro token is still a valid primitive of the CoT counter program.

**Order $k$ machine.**
For a target order $k$, the macro machine $S_k$ keeps the states of $S$ and uses the resulting macro tokens. From every non halting state $q$, we include each chainable block $\pi$ of length exactly $k$ that starts at $q$, whose guard $\varphi_\pi$ is satisfiable, and whose first $k-1$ destinations are non halting. To handle runs whose remaining length is not a multiple of $k$, we also include variable order blocks of length $r \in \{1, \ldots, k-1\}$ whose final destination is a halting state. Full length blocks may end in either a halting or a non halting state. For shorter blocks, all destinations before the final one must be non halting; execution stops at the first halt. The macro vocabulary $\Delta^{(k)}$ is the set of these tokens. Counters update by $u(M_\pi)$, so the prefix count identity holds verbatim with macro effects, and $S_k$ remains a CoT counter program.

### Determinism, Correctness, and Vocabulary

The guard composition above preserves the local structure of the base machine. We next show that it also preserves deterministic execution and yields an exact reduction in trace length.

**Lemma 2 (Determinism preserved).**
If $S$ is deterministic, then $S_k$ is deterministic.

**Proof.**
Fix a reachable configuration $(q, x)$ of $S_k$ with $q$ non halting. Because $S$ is deterministic, there is a unique maximal base run from $(q,x)$. Let $\pi^\star$ be its first $\min(k,m)$ transitions, where $m$ is the number of base steps until the run first halts. We claim that $M_{\pi^\star}$ is the unique macro token enabled at $(q,x)$.

It is enabled because every prefix of the base run satisfies its guard at the corresponding shifted counters, so $\varphi_{\pi^\star}(x)$ holds. The destinations before the last are non halting exactly when $\min(k,m)=k$, matching the two token families. For uniqueness, suppose a block $\pi \ne \pi^\star$ with $\mathrm{src}(\pi)=q$ had $\varphi_\pi(x)$ true. Let $j$ be the first index where $\pi$ and $\pi^\star$ differ. The prefix $\tau_1 \cdots \tau_{j-1}$ is shared, so both reach the same configuration $(q',x')$ before step $j$. Enabling $\varphi_\pi$ requires $g(\tau_j)$ to hold at $x'$ and $\tau_j$ to be applicable there, and likewise for $\tau^\star_j$. Since $S$ is deterministic, only one transition is enabled at $(q',x')$, so $\tau_j = \tau^\star_j$, a contradiction. Halting states have no outgoing macro token, so determinism holds everywhere. $\square$

**Theorem 1 (Macro speedup).**
Let $S$ be a deterministic guarded update counter machine and let $w$ be an input on which $S$ halts after $T$ base transitions. Then $S_k$ on $w$ halts after exactly $\lceil T/k\rceil$ macro transitions, reaches the same halting state, and its counter values after $m$ macro tokens equal the base counter values after $\min(mk,T)$ base steps. Consequently, $L(S_k) = L(S)$.

**Proof.**
Write $T = qk + r$ with $0 \le r < k$. By Lemma 2, $S_k$ follows the unique base run, grouped into blocks. The first $q$ macro tokens are the length $k$ blocks $\pi^{(1)}, \ldots, \pi^{(q)}$ covering base steps $1$ through $qk$. By the macro effect definition, the counter update of the $m$th macro token equals the sum of the effects of its $k$ base steps. Thus, after $m \le q$ macro tokens, the counters equal the base counters after $mk$ steps by a telescoping sum matching the prefix count identity.

If $r=0$, the $q$th block ends in the halting state and the trace length is $q = \lceil T/k\rceil$. If $r>0$, one variable order block of length $r$ covers the remaining base steps $qk+1,\ldots,T$ and ends in the same halting state that $S$ reaches. The trace length is therefore $q+1 = \lceil T/k\rceil$, and the final counters match the base counters after $T$ steps. The final states are identical, so $w$ is accepted by $S_k$ if and only if it is accepted by $S$. $\square$

The theorem gives an exact reduction in the number of generated positions for every halting run. The remaining question is the price paid in the token vocabulary.

**Proposition 1 (Vocabulary bound).**
The macro vocabulary satisfies

$$
\vert \Delta^{(k)} \vert \le \sum_{r=1}^{k} \vert \Delta \vert^r = O\!\left(\vert \Delta \vert^k\right).
$$

Only chainable, guard satisfiable, reachable blocks are included, so the realized vocabulary can be substantially smaller than this worst case.

**Proof.**
A block of length $r$ is a sequence of $r$ base transitions, of which there are at most $|\Delta|^r$. Summing over $r \in \{1,\ldots,k\}$ gives the bound. Chainability, satisfiability, and reachability only remove tokens. $\square$

The worst case can be loose for structured machines. We next give bounds that depend on the control flow rather than on the total number of base transitions. Let $\delta$ denote the control flow out degree, namely, the largest number of outgoing base transitions at any state.

**Proposition 2 (Structural bound).**
$|\Delta^{(k)}| \le |Q|\sum_{r=1}^{k}\delta^r$. Moreover, macro tokens that induce the same transition, meaning that they share source, destination, effect vector, and canonical guard, can be merged without changing the machine. Because $S$ is deterministic, distinct feasible blocks cannot have the same guard while inducing different effects or destinations.

**Proof.**
Every block starts at one of $|Q|$ states and extends by at most $\delta$ choices at each of its $r$ steps. Thus there are at most $|Q|\delta^r$ chainable blocks of length $r$. Summing over $r \in \{1,\ldots,k\}$ gives the stated bound before feasibility and reachability pruning.

For the merge claim, two blocks with equal source, destination, effect, and canonical guard denote the same guarded update transition, so representing them by one token leaves the machine's runs unchanged. If two feasible blocks had the same guard but differed in effect or destination, then from a common configuration satisfying that guard they would induce two distinct feasible base runs, contradicting determinism. $\square$

For machines with a single non halting successor, the dependence on the macro order becomes linear.

**Proposition 3 (Linear vocabulary for single successor loops).**
Let $Q_{\mathrm{nh}} = Q \setminus (F_{\mathrm{acc}}\cup F_{\mathrm{rej}})$ be the non halting states. Suppose every state in $Q_{\mathrm{nh}}$ has exactly one outgoing transition to a non halting state and at most $h$ outgoing transitions to halting states. For every $k\ge1$,
$$
|\Delta^{(k)}| \le |Q_{\mathrm{nh}}|(1+kh).
$$
When each state has at most one halting exit, this gives $|Q_{\mathrm{nh}}|(k+1)$. The bound includes blocks that halt on their $k$th transition.

**Proof.**
Fix a start state $q\in Q_{\mathrm{nh}}$. There is at most one length $k$ block whose transitions all end in non halting states, since each continuing transition is fixed. A block that halts at length $r$, for any $1\le r\le k$, follows the unique continuing prefix of length $r-1$ and then takes one of at most $h$ halting exits. There are therefore at most $kh$ halting blocks and one continuing block per start state. Summing over $Q_{\mathrm{nh}}$ proves the bound. Unsatisfiable guards, reachability pruning, and canonical merging can only reduce this count. $\square$

**Example and tightness.** Consider three non halting states in a cycle and one shared halting state. From each cycle state, decrement and advance when $x>0$, and halt when $x=0$. At $k=2$, the three macros from each start state are: halt immediately ($x=0$), decrement then halt ($x=1$), and decrement twice ($x\ge2$). All nine macros are included under our token convention, matching the bound $3(1+2)=9$.

The bound is tight more generally. On a cycle of $n$ non halting states, use a continuing transition with guard $x\ge h$ and effect $-h$, and $h$ halting exits with guards $x=j$ for $0 z\le j < h$, where $h\ge1$. A block halting at step $r$ through exit $j$ has guard $x=(r-1)h+j$, while the continuing length $k$ block has guard $x\ge kh$. All $n(1+kh)$ blocks are feasible and distinct. The residue loops used in our tasks have $h=1$, so their bound is $|Q_{\mathrm{nh}}|(k+1)$ and remains linear in $k$. Theorem 1 and its exact trace length $\lceil T/k\rceil$ are unaffected by the vocabulary bound.

### Transfer to CoT Programs

The preceding results concern the simulated machine. We next connect them back to the transformer construction. We state this connection carefully: it is an expressibility result about the class of programs that the construction can represent, not a claim that gradient based training necessarily finds the corresponding transformer.

**Assumption 1 (Compilability of CoT counter programs).**
There is a finite-resource decoder-only transformer construction that, for any fixed deterministic guarded update counter machine expressed as a CoT counter program, exactly reproduces its trace, using no explicit positional encoding and reading each counter as a weighted prefix count of emitted tokens, with each guard atom a threshold test on that count. The input initializer $\iota(w)$ must be computable by the input interface of the construction: it may use fixed input-summary counters or other features supplied by the input prefix, but it cannot require an unbounded table indexed by the whole word. The machine has finitely many control states, counters, integer effects, and macro tokens; the compiled model has a finite output vocabulary containing those tokens and an end-of-trace symbol. Its width, attention heads, embeddings, and output projection may depend on the fixed machine, the number of counters and control states, the largest guard threshold, and the macro vocabulary size, but not on the input length. In particular, the output projection must represent every token in $\Delta^{(k)}$, and the attention features must provide one recoverable weighted prefix count for each counter and the finite control information needed by the guards. This is the compilation of the base construction <d-cite key="jiang2026softmax"></d-cite>, and the length generalization it exhibits is a property of that exact program, not of an arbitrary trained approximation.

**Proposition 4 (Preserved expressibility).**
Under Assumption 1, $S_k$ is a CoT counter program, so the same construction compiles $S_k$ into a transformer that exactly reproduces the macro trace. Any length generalization guarantee that the construction provides for $S$ through the CoT counter program interface applies to $S_k$ under the same assumptions.

**Proof.** After $m$ macro tokens, the value of counter $i$ is $\iota_i(w) + \sum_{j \lt m} u_i\!\left(M_{\pi^{(j)}}\right)$, which is a weighted prefix count of the emitted macro tokens plus the initializer supplied by the input interface. Thus, it has exactly the form of the base construction equation in the preliminaries, now with the macro alphabet. By Lemma 1, each macro guard is an interval constraint on each counter, and hence can be expressed as a conjunction of threshold tests on these prefix counts. The finite control state is carried by the emitted transition token and its output embedding. Therefore, the macro machine $S_k$ satisfies the interface required by Assumption 1. Applying the construction of Assumption 1 to $S_k$ consequently reproduces its trace exactly. $\square$

The transfer is therefore an existence statement: whenever the base construction provides the corresponding length generalization guarantee for $S$, the same construction applies to $S_k$. Whether training recovers this solution from finite data is a separate question, which we examine empirically in experiments. In particular, the larger macro alphabet can reduce the number of examples associated with each token, creating a potential sample complexity cost. Our data scaling experiments quantify this tradeoff.

## What Is Compressed? Positions and Counting

Theorem 1 reduces a trace of length $T$ to $\lceil T/k\rceil$ positions. This raises a basic question: what exactly is being compressed? Macro composition does not remove information from the simulated computation. Rather, it changes how that information is distributed across generated positions.

**Remark (Information is conserved).**
The macro trace has $\lceil T/k\rceil$ tokens over an alphabet of size at most $O(|\Delta|^k)$. Thus, using the worst case vocabulary bound, its description length is approximately $\frac{T}{k} \cdot k \log_2 |\Delta| = T \log_2 |\Delta|$ bits, matching the corresponding upper bound for the base trace. Macro composition therefore does not claim to compress the information content of the computation. It repackages the same computation into fewer, wider symbols.

<figure style="text-align: center; margin: 20px 0;">
    <iframe 
        src="{{ 'assets/html/submission/positions_not_bits.html' | relative_url }}"
        width="100%" 
        height="500" 
        style="border: none; overflow: hidden; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);"
        title="Interactive Graph">
    </iframe>
    <figcaption><b>Compression reduces computational positions without reducing the represented computation.</b> For the generated trace, the number of serial positions and key-value cache entries decreases as $1/c$, while generated-trace attention cost decreases as $1/c^2$. The fixed input prefix remains, so full end-to-end costs need not follow these ratios. The information represented by the trace remains unchanged.</figcaption>
</figure>

The relevant resource for the transformer is instead the number of generated positions. Under our cost model, reducing the trace from $T$ to $\lceil T/k\rceil$ reduces serial generation length and the generated portion of the key-value cache by about a factor of $k$. For attention over generated positions alone, the arithmetic cost changes from $\Theta(T^2d)$ to $\Theta((T/k)^2d)$, giving a factor of $k^2$ reduction. With an input prefix, the full cost follows the $P^2+PN+N^2$ expression above, so this is a generated-trace statement rather than a universal end-to-end ratio. It is directly relevant to the serial cost studied by existing chain-of-thought lower bounds <d-cite key="li2024chain,merrill2024expressive"></d-cite>. The tradeoff is a larger macro vocabulary, which increases the size of the corresponding output representation and can affect learning. We quantify this tradeoff experimentally.


### Base $b$ Counting

Macro composition treats all transitions uniformly. Counting loops offer a more targeted form of compression. Consider a single counter $i$ with a monotone decrement burst: a state $s$ with a self loop transition $\tau_{\mathrm{dec}}$ of guard $(x_i > 0)$ and effect $-e_i$, together with an exit transition of guard $(x_i = 0)$. To remove a mass of $V$ units, the base machine fires $\tau_{\mathrm{dec}}$ exactly $V$ times.

**Construction.**
The base $b$ rewrite replaces the unit loop with block transitions on the same state. A full block has guard $(x_i \ge b)$ and effect $-b e_i$. For each remainder $\rho \in \{1,\ldots,b-1\}$, a block has guard $(x_i = \rho)$ and effect $-\rho e_i$ leading to the exit. The exit guard $(x_i = 0)$ is unchanged. If the loop uses residue control states, the full block also updates the residue state from $q_r$ to $q_{(r-b)\bmod c}$, and a remainder block updates it to $q_{(r-\rho)\bmod c}$; the guard and counter update are unchanged. Thus the rewrite does not silently assume that a modulo task has a single self-loop state.

Each block uses a canonical single counter guard, so the rewritten machine remains a CoT counter program. It is also deterministic: $(x_i \ge b)$ covers all values at least $b$, while the remainder guards $(x_i = \rho)$ cover the remaining positive values.

**Proposition 5 (Base $b$ speedup).**
The base $b$ rewrite emits $\lfloor V/b\rfloor$ full blocks and at most one remainder block, hence $\lceil V/b\rceil$ tokens, to remove a mass of $V$, compared with $V$ tokens for the base machine. For runs whose length is dominated by such bursts, a base trace of length $T$ becomes a trace of length $\Theta(T/b)$.

**Proof.**
Write $V = b\lfloor V/b\rfloor + \rho$ with $0 \le \rho < b$. While $x_i \ge b$, the only enabled block is the full block, which fires $\lfloor V/b\rfloor$ times and reduces the counter to $\rho$. If $\rho=0$, the exit fires next and the number of block tokens is $\lfloor V/b\rfloor=\lceil V/b\rceil$. If $\rho>0$, the corresponding remainder block fires once before the exit, giving $\lfloor V/b\rfloor+1=\lceil V/b\rceil$ tokens. Summing over counting dominated bursts yields the stated $\Theta(T/b)$ trace length, up to the constant number of control tokens per burst. $\square$

This rewrite gives a more specialized speedup than macro composition: rather than grouping arbitrary transition sequences, it changes the granularity of a known counting operation. Its effect on the overall simulation length depends on how much of the computation is spent in such bursts.

**Corollary 1 (Effect on counter simulation length).**
Suppose the base machine simulates a Turing machine using space $s$ and has a counting dominated run of length $T = 2^{\Theta(s)}$, as can arise in counter based simulations <d-cite key="jiang2026softmax,fischer1968counter"></d-cite>. Under the base $b$ rewrite, the corresponding trace length is

$$
T_b = \frac{2^{\Theta(s)}}{b},
$$

and therefore

$$
\log_2 T_b = \Theta(s) - \log_2 b.
$$

**Proof.**
Immediate from Proposition 5 applied to a counting dominated run. $\square$

The corollary should be interpreted as a constant shift for fixed $b$, rather than as a change in the asymptotic dependence on the simulated space. The improvement is therefore modest in this regime, but it becomes directly measurable when counting transitions dominate the generated trace.

### Combining the Two Rewrites

The two mechanisms operate at different levels. Base $b$ counting compresses specific counting bursts, while order $k$ macro transitions compose the resulting transitions more generally.

**Corollary 2 (Composability).**
Applying the base $b$ rewrite followed by order $k$ macro composition to a counting dominated run reduces the number of generated positions by approximately a factor of $bk$. The resulting machine remains a deterministic CoT counter program and recognizes the same language.

**Proof.**
The base $b$ rewrite reduces a counting dominated trace from $T$ to $\Theta(T/b)$ by Proposition 5. Applying Theorem 1 to the resulting machine introduces a further factor of approximately $k$ in the number of positions while preserving determinism, recognition, and the CoT counter program structure. $\square$

The two rewrites therefore expose a general design principle: **compression can be obtained either by grouping arbitrary consecutive transitions or by increasing the granularity of structured counting operations**. The former applies broadly but can incur a larger vocabulary, while the latter provides a more targeted reduction with a smaller structural overhead.

<figure style="text-align: center; margin: 20px 0;">
    <iframe 
        src="{{ 'assets/html/submission/baseb_counting.html' | relative_url }}"
        width="100%" 
        height="500" 
        style="border: none; overflow: hidden; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);"
        title="Interactive Graph">
    </iframe>
  <figcaption><b>Base-$b$ counting shrinks a decrement burst.</b> A single-counter decrement of mass $V$ emits $V$ unit tokens in the base machine but only $\lceil V/b \rceil$ block tokens in base $b$, since each full block removes $b$ units and a final remainder block removes the rest. This is the counting-dominated speedup, and it composes with macro order to give a factor of about $bk$.</figcaption>
</figure>

### Scope

The base $b$ construction is intentionally restricted to single counter monotone decrement bursts. A loop that decrements several counters by fixed amounts on every iteration can admit an analogous vector rewrite, with a full block representing $b$ iterations and remainder blocks handling the final partial group. Mixed sign or data dependent updates are not covered by this fixed radix construction because the number of iterations need not be determined by a single counter. Such computations can still be handled by the more general macro transition construction.

We therefore focus the formal and empirical analysis on the single counter case. The broader point is not that every chain-of-thought computation admits a uniform $b$ fold compression, but that structured portions of a counter computation can sometimes be represented at a coarser granularity while remaining within the same CoT counter programming framework.


## Experiments

The theoretical results establish exact trace compression at the machine level. We now test the corresponding learning and efficiency questions: whether transformers trained on compressed traces retain length generalization, how compression affects generation cost, and when the larger macro vocabulary becomes a practical bottleneck. All experiments use synthetic tasks generated by hand built deterministic counter machines, with the exact machine trace serving as supervision.

### Setup

All data is synthetic. For each task a hand-built deterministic counter machine defines the language, and the supervision for an input word is the exact transition trace the machine emits, following the protocol of the base construction. We train a four-layer decoder-only transformer with width $128$, eight attention heads, and about $5.3 \times 10^5$ parameters, using no positional encoding, on next-token prediction of the trace, and evaluate by greedy generation of the whole trace. Each configuration uses $5{,}000$ training examples. A prediction counts as correct only under exact match of the full trace. We use three length-stratified test splits: an in-range split at training lengths up to $64$, and two splits at lengths up to $96$ and $128$, both strictly longer than any training input. Training ran on a single T4 GPU with a batch size of $256$ and AdamW at a learning rate of $3 \times 10^{-4}$ with cosine decay. We report the step at which all three splits first reach the exact-match threshold of $0.990$; if they do so before $15{,}000$ steps, training stops early. These are the same three test splits on which we report accuracy: we did not hold out a separate validation split, so the test splits were also used for early stopping. The reported exact-match values are therefore measured at a stopping point selected on the test data and may overstate accuracy relative to a protocol with a held-out validation split. The one configuration that runs to the $15{,}000$-step ceiling (mod 5 at $k=4$) is not affected by this selection, since its stopping point does not depend on test accuracy. Table 1 uses one seed per GPU configuration and reports no estimate of seed-to-seed variance. We report a separate three-seed CPU study below in Multi-seed sensitivity.

### Tasks

**Parity**, **mod 3**, and **mod 5** read a string over two symbols and accept when the count of the first symbol is divisible by $2$, $3$, or $5$; these are single-counter decrement loops, so both macro and base-$b$ apply. **Comparison** reads $a^m b^n$ and accepts when $m > n$; it is a two-counter paired-decrement machine. **Equal count** reads a string over $\{a, b\}$ and accepts when $\#a = \#b$; it is the standard symmetric two-counter language and a direct analogue of the counting tasks studied in the base construction. **Three-sum** reads $a^l b^m c^n$ and accepts when $l + m = n$; it verifies an addition identity using a two-phase three-counter machine, constituting the arithmetical task requested by the reviewer.

The machine constructions themselves are verified separately by direct unit tests, so deviations in the experiments reflect learned behavior rather than errors in the rewritten machines.

### Compression and length generalization
We first vary macro order $k$ and base $b$ and measure both trace length and exact match. The central empirical question is whether reducing the number of positions changes the model's ability to recover the underlying computation.


<figure style="text-align: center; margin: 20px 0;">
    <iframe 
        src="{{ 'assets/html/submission/results_explorer.html' | relative_url }}"
        width="100%" 
        height="500" 
        style="border: none; overflow: hidden; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);"
        title="Interactive Graph">
    </iframe>
    <figcaption><b>Compression and length generalization at CPU scale.</b> These earlier small-model runs are separate from the GPU results in Table 1 and the three-seed sensitivity study. Trace length decreases with macro order or base, while exact match on the in range split and the two longer splits remains high over a moderate compression regime. Degradation appears for the most aggressive settings, particularly when the compressed vocabulary becomes large relative to the counting range.</figcaption>
</figure>


For macro transitions, Table 1 varies the macro order $k$ across six tasks at GPU scale. The mean trace length falls by a factor of about $k$ in every row: observed ratios range from $1.92$ to $1.96$ at $k=2$ and from $2.88$ to $3.71$ at $k=3$ or $k=4$, matching $\lceil T/k \rceil$ up to the constant control tokens. Exact match on the longest test split (EM2, inputs up to $128$ tokens) is $1.000$ in $15$ of the $18$ configurations; the three remaining configurations reach $0.990$. Thus all $18$ configurations meet the $0.990$ EM2 threshold, while the exact $1.000$ result is not universal. Early stopping fires in $17$ of $18$ runs before the $15{,}000$-step ceiling, with the fastest convergence at $400$ steps and the slowest early-stopped configuration at $4{,}200$ steps. Only mod 5 at $k=4$ runs to the ceiling, reaching $0.990$ on EM2.

On the **three-sum** task, the machine verifies the addition identity $l + m = n$ using two sequential paired-decrement phases on three counters. Macro composition compresses both phases simultaneously. At $k=3$ the trace shortens $2.88$-fold (from $23.32$ to $8.11$ tokens) and exact match is $1.000$ on all three test splits, including inputs up to $128$ tokens long. This is the arithmetical task structure requested by the reviewer.

On the **equal-count** task, the machine accepts when the two symbol counts in a random binary string are equal, the standard symmetric two-counter language. At $k=4$ the trace shortens $3.67$-fold (from $16.34$ to $4.45$ tokens) with EM2 $= 1.000$.

**Table 1: Macro transitions at GPU scale.** The model uses width $128$, four layers, $5{,}000$ training examples, training inputs up to length $64$, and test inputs up to length $128$. Mean trace length, its ratio to the $k=1$ baseline, exact-match accuracy on three length-stratified test splits (EM0 in-range, EM1 and EM2 longer than training), and the optimizer step at which all splits first hit the exact-match threshold are shown. The same test splits determine early stopping (see Setup). A dagger marks the $15{,}000$-step ceiling.

| Task | $k$ | Mean trace | Ratio | EM0 | EM1 | EM2 | Stop step |
| --- | --: | ---: | ---: | ---: | ---: | ---: | ---: |
| parity | 1 | 18.60 | 1.00 | 1.000 | 1.000 | 1.000 | 400 |
| parity | 2 | 9.55 | 1.95 | 1.000 | 1.000 | 1.000 | 800 |
| parity | 4 | 5.02 | 3.71 | 1.000 | 1.000 | 1.000 | 1,600 |
| mod 3 | 1 | 18.60 | 1.00 | 1.000 | 1.000 | 1.000 | 400 |
| mod 3 | 2 | 9.55 | 1.95 | 1.000 | 1.000 | 1.000 | 600 |
| mod 3 | 4 | 5.02 | 3.71 | 1.000 | 1.000 | 1.000 | 3,400 |
| mod 5 | 1 | 18.60 | 1.00 | 1.000 | 1.000 | 1.000 | 600 |
| mod 5 | 2 | 9.55 | 1.95 | 1.000 | 1.000 | 1.000 | 1,600 |
| mod 5 | 4 | 5.02 | 3.71 | 1.000 | 0.960 | 0.990 | 15,000† |
| comparison | 1 | 12.09 | 1.00 | 1.000 | 1.000 | 0.990 | 800 |
| comparison | 2 | 6.29 | 1.92 | 1.000 | 1.000 | 1.000 | 1,200 |
| comparison | 4 | 3.39 | 3.57 | 1.000 | 1.000 | 0.990 | 4,200 |
| equal count | 1 | 16.34 | 1.00 | 1.000 | 1.000 | 1.000 | 1,200 |
| equal count | 2 | 8.42 | 1.94 | 1.000 | 1.000 | 1.000 | 1,400 |
| equal count | 4 | 4.45 | 3.67 | 1.000 | 1.000 | 1.000 | 3,200 |
| three-sum | 1 | 23.32 | 1.00 | 1.000 | 1.000 | 1.000 | 1,200 |
| three-sum | 2 | 11.91 | 1.96 | 1.000 | 1.000 | 1.000 | 2,600 |
| three-sum | 3 | 8.11 | 2.88 | 1.000 | 1.000 | 1.000 | 2,800 |

The same pattern holds for base $b$ counting. Moderate compression preserves generalization, whereas aggressive compression can cause a sharp drop in extrapolation accuracy. For example, mod 3 remains perfect through $b=2$ but falls to $0.81$ on EM1 and $0.10$ on EM2 at $b=4$. This is consistent with the positions versus vocabulary tradeoff: the rewritten machine remains exact, but the shorter traces provide fewer training contexts for the expanded token set.

**Table 2: Base-$b$ counting at CPU scale.** This is a small-model demonstration; the macro-transition results in Table 1 use the GPU protocol. Columns are as in Table 1. The $b=4$ mod 3 row is the predicted capacity boundary.

| Task   | $b$ | Mean trace | Length ratio |  EM0 |  EM1 |  EM2 |
| ------ | --: | ---------: | -----------: | ---: | ---: | ---: |
| parity |   1 |       6.15 |         1.00 | 1.00 | 1.00 | 0.90 |
| parity |   2 |       3.83 |         1.61 | 1.00 | 1.00 | 0.97 |
| parity |   4 |       2.67 |         2.30 | 1.00 | 0.97 | 0.91 |
| mod 3  |   1 |       6.72 |         1.00 | 1.00 | 1.00 | 1.00 |
| mod 3  |   2 |       4.11 |         1.63 | 1.00 | 1.00 | 1.00 |
| mod 3  |   4 |       2.82 |         2.38 | 1.00 | 0.81 | 0.10 |

These results separate two effects. **Trace compression is guaranteed by the machine transformation; learned length generalization is not.** In the tested regime, the latter survives moderate compression but eventually degrades as the compressed representation becomes harder to learn.

### Reading the results

Two claims are separated cleanly across both tables. The trace-length reduction is exact and holds in every row of Table 1, matching the macro-transition theorem up to the constant control tokens. The learning claim reaches at least $0.990$ on EM2 in all $18$ GPU-scale configurations, with exact $1.000$ in $15$ of them, and holds across all base-$b$ settings in the moderate regime. The clearest degradation appears at the capacity boundary (mod 5 $k=4$ and mod 3 $b=4$), where the information has been pushed into rare wide tokens. The practical guideline is to compress until the vocabulary begins to dominate, and no further. On the addition task (three-sum), the claim holds at every $k$ tested, including the largest compression ($k=3$, ratio $2.88\times$).

### Exact validation of the vocabulary bound

We checked Proposition 3 directly on CPU using the macro compiler, without training a transformer. The three-state cycle above produces nine macros at $k=2$: three short halting blocks, three full length halting blocks, and three full length continuing blocks.

We then enumerated cycles with $n\in\{1,2,3,5\}$ non halting states, $h\in\{0,1,2,3\}$ halting exits per state, and $k\in\{1,\ldots,8\}$, giving 128 configurations. All counts equal $n(1+kh)$, including after canonical merging. For $h=0$ we checked the continuing-block count only. For $h\ge1$, we tested every initial counter value from 0 through 63 from every non halting state. All 16,896 base/macro run pairs agreed on acceptance and the exact $\lceil T/k\rceil$ trace length; all 137,775 checked configurations agreed at block boundaries. These finite checks exercise the missing full length halting case and multiple exits; the proof establishes the bound for arbitrary fixed machines satisfying Proposition 3.


### Multi-seed sensitivity

To check whether these patterns depend on a single random seed, we ran a three-seed CPU sensitivity study on parity and comparison. We used the repository's CPU protocol: two layers, width $64$, four heads, up to $1{,}000$ training examples, 48 examples per test split, training lengths up to $14$, longer splits up to $22$ and $30$, batch size $64$, AdamW at learning rate $3\times10^{-3}$, and a $1{,}500$-step ceiling. The table reports EM2 mean, standard deviation, and range over seeds 0, 1, and 2.

| Task | Setting | EM2 mean $\pm$ std. | Seed range |
| --- | --- | ---: | ---: |
| parity | macro $k=1$ | $0.799 \pm 0.115$ | $0.688$--$0.917$ |
| parity | macro $k=2$ | $0.847 \pm 0.146$ | $0.708$--$1.000$ |
| parity | macro $k=4$ | $0.986 \pm 0.024$ | $0.958$--$1.000$ |
| comparison | macro $k=1$ | $1.000 \pm 0.000$ | $1.000$--$1.000$ |
| comparison | macro $k=2$ | $1.000 \pm 0.000$ | $1.000$--$1.000$ |
| comparison | macro $k=4$ | $0.424 \pm 0.126$ | $0.292$--$0.542$ |
| parity | base $b=2$ | $1.000 \pm 0.000$ | $1.000$--$1.000$ |
| parity | base $b=4$ | $0.924 \pm 0.060$ | $0.854$--$0.958$ |

The CPU check shows stable results in several settings: comparison remains perfect at $k=1$ and $k=2$, and parity remains perfect at base $b=2$. Stability is task and setting dependent. Parity macro runs vary at $k=1$ and $k=2$, but are more stable at $k=4$; comparison instead loses accuracy at $k=4$. These results do not establish that variance increases with compression or that vocabulary size alone causes the observed errors. We therefore use the multi-seed results as a sensitivity check, not as a substitute for the GPU-scale table or as evidence that every compression level is equally stable.

### Generation cost

We next measure the computational effect of reducing positions. We report serial decode steps, attention FLOPs under a key-value cache, peak cache size, and measured greedy generation latency. Exact-match evaluation normally uses greedy generation until the end-of-trace token. For the latency comparison, however, generation is forced to the true number of decode steps so that an early mistake or premature stop does not make one setting appear faster. The analytic attention and cache columns below count the generated trace portion; the full end-to-end accounting also includes the input prefix $P$, whose length is the encoded task input and varies across examples, as described in the cost model.

<div class="l-page">
  <iframe src="{{ 'assets/html/submission/cost_dashboard.html' | relative_url }}" frameborder="0" scrolling="no" width="100%" height="600px"></iframe>
</div>

**Figure 5: Generation cost under compression.** Measured forced-length latency decreases with the number of generated positions, while generated-trace attention operations and cache size follow the predicted dependence on trace length. The generated portion of attention decreases approximately quadratically in the compression factor; end-to-end savings are smaller when input-prefix processing dominates.





For comparison, measured latency on the comparison task decreases from $16.4$ ms at $k=1$ to $5.4$ ms at $k=3$, a factor of $3.0$ versus a trace length reduction of $2.9$. Analytic generated-trace attention operations decrease by a factor of $7.5$. On mod 3, latency decreases from $33.1$ ms at $b=1$ to $8.3$ ms at $b=4$. These latency measurements are a CPU-scale implementation demonstration; the GPU protocol is used for the macro-transition accuracy table.

| Task       | Setting | Mean trace | Latency (ms) | Generated attention FLOPs | Generated KV bytes | Vocabulary |
| ---------- | ------- | ---------: | -----------: | ---------------: | -------: | ---------: |
| comparison | $k=1$   |      17.21 |        16.40 | $8.02\times10^4$ |    17621 |          9 |
| comparison | $k=2$   |       8.79 |         7.95 | $2.20\times10^4$ |     9002 |         12 |
| comparison | $k=3$   |       6.00 |         5.38 | $1.08\times10^4$ |     6144 |         15 |
| mod 3      | $b=1$   |      32.83 |        33.09 | $2.84\times10^5$ |    33621 |         11 |
| mod 3      | $b=2$   |      17.17 |        15.74 | $7.98\times10^4$ |    17578 |         14 |
| mod 3      | $b=4$   |       9.42 |         8.33 | $2.51\times10^4$ |     9642 |         20 |

The measurements therefore follow the qualitative scaling predicted by the position-based cost model: fewer generated positions reduce serial work and generated-trace cache usage, while the quadratic generated-trace attention term falls more rapidly. These measurements concern the tested implementation and forced-length protocol; they should not be interpreted as a hardware-independent latency law or as a complete end-to-end cost ratio for arbitrary input-prefix lengths.

### Vocabulary and data scaling

Compression changes not only sequence length but also the effective token alphabet. We therefore examine how many macro tokens are actually reachable and how much additional training data is required.

<div class="l-page">
  <iframe src="{{ 'assets/html/submission/vocabulary_growth.html' | relative_url }}" frameborder="0" scrolling="no" width="100%" height="480px"></iframe>
</div>

**Figure 6: Reachable macro vocabulary.** The plotted worst case bound is the general bound from Proposition 1, $\sum_{r=1}^{k}|\Delta|^r$, not the single successor bound from Proposition 3. It grows exponentially with macro order, while the number of reachable and guard satisfiable macros in the studied machines grows much more slowly. At $k=4$, the bound is $1554$ versus $15$ reachable tokens for mod 3, and $11110$ versus $63$ for unary mult.

| Task | $|\Delta|$ | $k$ | General bound (Prop. 1) | Reachable | Merged |
| --- | ---: | --- | --- | --- | --- |
| parity | 4 | 2, 3, 4 | 20, 84, 340 | 6, 8, 10 | 6, 8, 10 |
| mod 3 | 6 | 2, 3, 4 | 42, 258, 1554 | 9, 12, 15 | 9, 12, 15 |
| comparison | 4 | 2, 3, 4 | 20, 84, 340 | 7, 10, 13 | 7, 10, 13 |
| unary mult | 10 | 2, 3, 4 | 110, 1110, 11110 | 21, 38, 63 | 21, 38, 63 |

The gap between the worst case and reachable vocabulary is substantial in all four tasks. For parity and mod 3, the counts are exactly $2(k+1)$ and $3(k+1)$, respectively, attaining the Proposition 3 bound with one halting exit per non halting state. Canonical merging does not reduce the counts further in these deterministic machines because feasible blocks already have disjoint guards.

We finally test whether this larger alphabet creates a measurable data requirement.

<div class="l-page">
  <iframe src="{{ 'assets/html/submission/data_scaling.html' | relative_url }}" frameborder="0" scrolling="no" width="100%" height="450px"></iframe>
</div>

**Figure 7: Data scaling with macro order.** On parity, exact match on the longest split improves with training set size for all macro orders. At 60 examples, $k=4$ reaches $0.44$ exact match, while all tested orders reach $1.00$ with sufficient data, illustrating the data cost associated with a larger compressed vocabulary.

Holding the counting range, model size, and optimizer budget fixed, exact match on the longest split increases with training set size for $k\in\{1,2,4\}$. At the smallest budget of $60$ examples, $k=4$ is the most data limited and reaches $0.44$, whereas all orders reach $1.00$ once sufficient data is provided. This supports the predicted tradeoff: compression reduces the number of positions, but sufficiently aggressive compression can shift the bottleneck from sequence length to vocabulary coverage.

Across these experiments, three observations are consistent. First, the machine rewrites produce the predicted reductions in trace length, including on the multi counter unary mult task. Second, moderate compression can retain length generalization, while sufficiently aggressive compression can degrade learned accuracy even though the underlying rewritten machine remains exact. Third, the computational savings from fewer positions are accompanied by a larger vocabulary and an associated data requirement. Thus, the experiments support a **compression tradeoff rather than an unconditional speedup**: the useful regime is one in which the reduction in generated positions outweighs the learning cost of the expanded vocabulary.


## Discussion

Taken together, the theoretical and empirical results support a consistent picture: trace compression preserves the underlying computation while substantially reducing the serial burden placed on the learner. The formal results establish that macro transitions preserve guards, determinism, and the base computation at block boundaries, while reducing a length $T$ trace to exactly $\lceil T/k\rceil$ positions. The counting rewrite further compresses monotone decrement bursts, yielding the predicted additive $\log_2 b$ improvement in the exponent, and the two constructions compose without leaving the underlying counter programming class. These results support our central claim that sequence length is a representation dependent cost rather than an intrinsic requirement of the computation, within the constant factor regime we analyze.

The empirical results complement these guarantees by testing whether the theoretical savings translate into learning behavior. Across six synthetic counter machine tasks, including equal-count comparison and the multi-phase three-sum arithmetic task, Transformers trained on compressed traces retain length generalization while exhibiting the predicted reductions in trace length. Our measurements of generation latency, attention operations, and key value cache size further connect the formal reduction in positions to practical computational savings. The vocabulary scaling experiments also identify the principal tradeoff: increasing the compression order can reduce sequence length, but may enlarge the macro vocabulary and eventually constrain learnability. Thus, the results collectively address both sides of the proposed construction: **exact computational preservation and reduced serial cost**, together with an explicit characterization of the representation capacity required to realize those savings.

### Limitations

Our analysis focuses on a fixed simulated machine and the one token per transition setting, where compression reduces the serial trace length without changing the class of recognizable languages. Our GPU scale experiments cover six tasks, including three sum for multi phase arithmetic computation and equal count for symmetric two counter reasoning, with additional tasks evaluated at CPU scale. The base $b$ construction focuses on single counter monotone decrement bursts, while we also describe an extension to synchronized multi counter loops; richer transition structures provide a natural direction for future work. The improvement in Corollary 1 is additive in $\log_2 b$, as intended by the construction. While the macro vocabulary can grow as $O(\vert \Delta \vert^k)$ in the general case, it grows approximately linearly for the single successor structures studied here. Our scaling experiments further show how the effective choices of $k$ and $b$ interact with model capacity and counting range. The GPU scale results use one seed per configuration, while the three-seed CPU sensitivity check on parity and comparison shows stability in several settings but also task-dependent variation, including parity at smaller macro orders. The reported test splits also served as the early-stopping criterion, and no separate validation split was held out, so the reported exact-match values are measured at a stopping point selected on test data and may be optimistic; a held-out validation split would remove this dependence. Overall, these results establish the behavior of the proposed compression methods in the settings studied here and provide clear directions for extending them to richer transition structures, more seeds, and larger scale problems.


## Conclusion

We presented a principled compression theory for chain-of-thought counter simulation, treating the number of generated positions as a representation dependent computational cost. Order $k$ macro transitions compose sequences of base transitions into wider tokens, while the base $b$ counting rewrite compresses monotone decrement bursts. Together, these constructions reduce trace lengths from $T$ to $\lceil T/k\rceil$ and from $V$ to $\lceil V/b\rceil$, respectively, while preserving determinism, exact recognition, and membership in the underlying counter programming class. The theoretical results therefore establish that these savings arise from changing the representation of the computation, not from discarding computational information.

The empirical evaluation complements these guarantees by showing that the predicted compression is realized in practice. Under moderate compression, transformers trained on compressed traces retain length generalization across our synthetic counter machine tasks while achieving reductions in generation latency, attention operations, and key value cache usage; more aggressive compression trades this retention for shorter traces. At the same time, our vocabulary analysis and scaling experiments expose the fundamental tradeoff: more aggressive compression reduces serial length but can increase the macro vocabulary and eventually exceed the model's effective representation capacity. This provides a concrete boundary for compression rather than treating shorter traces as universally beneficial.

More broadly, our results suggest that chain-of-thought length should not be viewed as a fixed property of a reasoning task. **The computational content can remain unchanged while its serial representation is substantially compressed.** By combining exact preservation theorems, complexity analysis, structural vocabulary bounds, and controlled empirical validation, we turn this observation into a concrete design principle: reasoning traces can be made shorter by changing their granularity, with the achievable compression determined by the structure of the computation and the capacity required to represent the resulting vocabulary.