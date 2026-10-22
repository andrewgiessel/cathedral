# Efficient Non-NUTS Inference for Cathedral

## Background, method framework, and a staged development proposal

**Status:** Proposal; implementation status is summarized below.\
**Baseline date:** 30 July 2026\
**Scope:** Gradient-free and non-Hamiltonian inference for expressive probabilistic programs

> **Implementation status:** Cathedral now includes `adaptive_mh` with warmup adaptation, support-aware scalar proposals, user-declared continuous blocks, and optional one-site prior resimulation on its documented fixed-structure subset. `infer_chains` runs independent sequential adaptive-MH chains and reports rank-normalized split/folded R-hat, ESS, and MCSE for scalar estimands. These are partial implementations of ideas in this proposal, not a general proposal/kernel framework. Gibbs and slice kernels, annealed SMC, programmable structural moves, parallel tempering, incremental execution, and learned proposals remain proposals. Section 3 records the review's original baseline; subsequent roadmap discussion remains design guidance, not a claim of implemented features.

## Abstract

Cathedral is a trace-based probabilistic programming library. It supports ordinary Python models, stochastic control flow, recursion, and memoization. It also supports mixed choices and traces that change structure. These features make Cathedral more expressive than a fixed-dimensional statistical model. They also limit methods that require one differentiable parameter vector.

This review examines methods that fit Cathedral's execution model. It recommends a common system for proposals and transition kernels, not one universal sampler. The system should support local methods such as adaptive Metropolis proposals, finite Gibbs updates, slice sampling, and correlated block moves. It should also support annealed Sequential Monte Carlo (SMC) and programmable structural moves.

The original proposed development order (not a completion checklist) is:

1. Harden trace identity, score accounting, diagnostics, and replay rules.
2. Add general proposal and kernel interfaces.
3. Add support-aware local proposals with warmup-only adaptation.
4. Add finite Gibbs, scalar slice, and user-defined blocks.
5. Add annealed Sequential Monte Carlo over complete traces.
6. Add targeted structural proposal programs.
7. Add parallel tempering, sequential particle methods, incremental execution, and learned proposals only when representative workloads justify them.

This order improves efficiency while preserving the features that make Cathedral distinct.

## 1. Purpose and design thesis

The No-U-Turn Sampler (NUTS) efficiently explores a fixed set of continuous variables. It uses gradients to move through a continuous posterior. NUTS is therefore not a universal engine for Cathedral. Cathedral models can create different random choices on different executions. They can also contain discrete branches, recursion, `mem()`, and `DPmem()`. The resulting posterior is over program traces, not one fixed-dimensional vector.

This review therefore asks a different question:

> How can Cathedral make inference much more efficient without giving up general Python or variable trace structure?

The scope is sampling-based inference. The review does not attempt to cover every method that is not NUTS. Score-function variational inference, deterministic approximation, and optimization are separate topics. Pseudo-marginal methods appear only where they connect directly to particle inference.

The answer has three parts.

First, Cathedral should improve local movement. The current Metropolis-Hastings (MH) engine selects one random choice and redraws it from its prior. This rule is general, but it can be inefficient. A local proposal can use the support and scale of the current choice. A block proposal can update correlated choices together. Gibbs and slice kernels can remove unnecessary rejection for eligible choices.

Second, Cathedral should add population methods. Annealed Sequential Monte Carlo can maintain many weighted traces and move them from the prior toward the posterior. Parallel tempering can run several trace chains at fixed temperatures and exchange complete traces. These methods need no global gradient and can operate on mixed or variable-dimensional traces.

Third, Cathedral should improve structural movement. The current engine can accept traces with different sets of choices. However, one prior redraw must cause each structure change. Programmable proposals, reversible-jump methods, and involutive Markov chain Monte Carlo (MCMC) can define deliberate structural moves.

The design thesis is:

> Cathedral should become a programmable system of valid trace kernels, not a collection of unrelated `method=` strings.

This thesis follows Church, Venture, Anglican, WebPPL, and Gen. It also fits Cathedral's current intervention-based replay model.

## 2. A common framework

### 2.1 The state is a trace

This review uses these terms consistently:

- A **choice** is one realized random variable in a trace.
- An **address** identifies that choice in the trace.
- A **site** is a source-code location that calls a sampling primitive.
- A **variable** is a mathematical quantity in the model.
- The **support** is the set of permitted values for a choice.
- A **replay-safe model** records all randomness and gives the same semantic address to the same choice.
- A **structural move** changes the set of addresses in a trace.

Let \(\tau\) denote one completed execution trace. Let \(A(\tau)\) be its set of choice addresses. Each address \(a\) has a value \(x_a\). It also has a conditional prior density or probability mass \(p_a\).

For a replay-safe model, define \(C(\tau)=1\) when all hard conditions succeed. Define \(C(\tau)=0\) otherwise. Let \(\ell(\tau)\) be the sum of finite scores from `observe()` and `factor()`. The unnormalized target density is:

$$
\widetilde{\pi}(\tau)=\left[\prod_{a \in A(\tau)}p_a(x_a\mid\text{execution history before }a)\right]C(\tau)\exp(\ell(\tau)).
$$

Traces can have different address sets and dimensions. Their state space is therefore a disjoint union of trace spaces. A valid method must operate on this full state space or state which fixed structures it supports.

This expression also assumes that all stochastic decisions are recorded and that addresses identify the same semantic choices during replay. Hidden randomness, duplicate names, and address aliasing violate that assumption.

### 2.2 Proposal, kernel, and inference algorithm

These terms describe different layers.

A **proposal** suggests a candidate move. For example, a proposal can add a Normal offset to a real value. It can also flip a Boolean value or propose a new program branch.

A **transition kernel** makes one statistically valid transition. A Metropolis kernel selects a proposal, computes the forward and reverse proposal probabilities, reruns the model, and accepts or rejects the candidate.

An **inference algorithm** applies one or more kernels many times. A Markov chain repeatedly applies kernels to one trace. SMC applies mutation kernels to a population of weighted traces.

A useful software contract is:

```python
class Kernel:
    def initialize(self, trace, target, rng):
        ...

    def step(self, trace, target, state, rng):
        # Return next_trace, next_state, and diagnostics.
        ...
```

An MH proposal needs a more specific result:

```python
class ProposalMove:
    interventions: dict
    log_q_forward: float
    log_q_reverse: float
    auxiliary_state: object | None
```

`log_q_forward` and `log_q_reverse` are complete proposal log densities for the target trace measure. Each includes density changes from support transformations. Use a separate input-to-output Jacobian only for a deterministic structural or involutive map. The kernel defines the map direction and adds the Jacobian once.

The kernel, not the proposal, owns the acceptance rule. Gibbs and slice kernels can implement the same `Kernel` interface without using an MH proposal.

### 2.3 Correctness comes before efficiency

For a move from \(\tau\) to \(\tau'\), the general MH acceptance probability is:

$$
\alpha(\tau,\tau')=1\wedge\frac{\widetilde{\pi}(\tau')q(\tau\mid\tau')}{\widetilde{\pi}(\tau)q(\tau'\mid\tau)}.
$$

For a structural move, \(q\) must include more than the new value at one address. It can include:

- the probability of selecting the address or move type;
- the probability of all newly created choices;
- the reverse probability of deleted choices;
- auxiliary random values used by the proposal;
- a Jacobian for a deterministic continuous transformation.

This bookkeeping is the main reason to centralize proposals and kernels. Custom inference code that omits one term can run normally while targeting the wrong posterior.

### 2.4 Efficiency has several meanings

Acceptance rate is useful, but it is not the final objective. A chain can accept many tiny moves and still explore slowly.

Cathedral should measure:

- effective sample size (ESS) from chain autocorrelation, per second;
- MCMC ESS per model execution;
- Monte Carlo standard error (MCSE);
- rank-normalized \(\widehat{R}\) across chains;
- expected squared jump distance for local kernels;
- mode occupancy and mode-switch frequency;
- structure-switch frequency;
- particle effective sample size and ancestor diversity;
- wall time, model executions, and memory.

MCMC ESS and weight-based ESS measure different properties. MCMC ESS measures autocorrelation in a Markov chain. Importance and particle ESS measure concentration in normalized weights.

For variable structures, report address presence and path frequency over the full chain. Do not remove iterations in which an address is absent before computing ESS or \(\widehat{R}\). That action changes the time series. Conditional value summaries remain useful, but they are not standard chain diagnostics without a defined full-chain estimand.

These measurements should use a benchmark set with finite discrete models, concentrated continuous posteriors, correlated blocks, multimodal targets, changing trace structures, recursion, and rare hard constraints.

### 2.5 Keep four concerns separate

The literature repeatedly separates four concerns:

1. **Trace semantics:** What is the state, and what density does it have?
2. **Kernel design:** How does one valid transition move through that state space?
3. **Algorithm composition:** How are kernels cycled, mixed, tempered, or applied to particles?
4. **Execution efficiency:** How much model code must rerun after a local change?

An incremental runtime can make each step faster without improving mixing. A good proposal can improve mixing while still rerunning the full model. Both improvements matter, but they solve different problems.

## 3. Original baseline (as of the proposal date)

This section records the state of Cathedral when this proposal was written; it is not a description of the integrated revision. The proposal's historical baseline was:

- `cathedral/trace.py` records `Trace` and `Choice` objects.
- `cathedral/primitives.py` provides `sample()`, `observe()`, `factor()`, `condition()`, `mem()`, and `DPmem()`.
- `cathedral/model.py` dispatches inference and returns a `Posterior`.
- `cathedral/inference/` contains rejection, importance, single-address MH, and exact finite enumeration.
- Interventions let an engine rerun a model while it holds selected choices fixed.
- The MH acceptance rule includes corrections for choices that appear or disappear. These corrections require stable addresses and fully traced replay.

The original `mh` step was intentionally simple:

1. Select one address uniformly.
2. Hold all other current choices fixed.
3. Let the selected choice redraw from its prior.
4. Rerun the complete model.
5. Accept or reject with a structural correction.

This gives Cathedral a broad fallback kernel. It also creates several bottlenecks.

- `Distribution.support()` cannot distinguish continuous, countably discrete, bounded, simplex, and vector domains.
- There is no general proposal density or reverse-proposal calculation.
- There is no kernel composition or block scheduling.
- The original baseline had no adaptive local-MH method; diagnostics exposed one run-level MH acceptance rate.
- `log_score` combines all observations and factors into one value.
- Auto-addresses depend on execution order. Duplicate explicit names overwrite prior entries.
- Every proposal reruns the complete Python function.
- `DPmem()` uses internal random decisions that are not separate trace choices.
- A `Posterior` stores samples. `Posterior.extend()` can continue MH from the last trace. It does not preserve random number generator (RNG) or adaptation state.
- The original baseline had no built-in multi-chain runner, \(\widehat{R}\), or Monte Carlo standard-error API.

The proposed general kernel system was initially intended to require unique explicit addresses and replay-safe models. The implemented `adaptive_mh` also requires unique explicit addresses and replay-safe, fixed-structure traces, and currently supports scalar numeric choices (with continuous-only blocks). Cathedral can relax this rule after it supports hierarchical identities. Loops, recursion, and memoized calls require identities that include semantic scope. The current `scope_path` is only visualization metadata.

Replay-based inference also requires controlled execution. Raw randomness, file I/O, wall-clock time, hidden mutation, and uncontrolled external state can invalidate replay.

## 4. The first priority: a proposal and kernel system

### 4.1 Support-aware proposal families

The mathematical support of a choice determines a safe local proposal family.

- A real value can use a Normal or Student-t random walk.
- A positive value can move in log space.
- A bounded value can move in scaled-logit space.
- A Boolean value can flip.
- A finite category can select another category.
- A nonnegative integer can use a reflected integer or discrete-Laplace move.
- A simplex can use an additive-log-ratio or Dirichlet drift move.
- A structural selector should retain a prior-resimulation component.

The proposal family and its scale are separate decisions. Support metadata selects the family. Warmup behavior selects the scale.

A robust default is a mixture:

$$
q(\tau'\mid\tau)=\rho q_{\mathrm{local}}(\tau'\mid\tau)+(1-\rho)q_{\mathrm{prior}}(\tau'\mid\tau).
$$

The local component gives efficient movement near the current posterior region. The prior component preserves broad support and can discover distant branches. The MH ratio must evaluate the complete mixture density in both directions.

The proposal state should live in the sampler. It should not live in a `Distribution` object because Cathedral reconstructs distributions during each execution. A distribution should expose domain metadata, event shape, bounds, finite support, and an optional transform. A proposal registry should use that metadata to select a default proposal.

A transformed random walk requires explicit density accounting. Symmetry in \(z=T(x)\) usually does not give symmetry in \(x\). Cathedral must account for the transformation in one place. It can evaluate the induced proposal density in \(x\), including its Jacobian. It can instead evaluate the target in transformed coordinates, including the inverse-transform Jacobian. It must not apply both corrections.

### 4.2 Warmup-only scale adaptation

The prior scale is often much wider than the posterior scale. A fixed default therefore cannot work well for all models.

Cathedral can adapt the log proposal scale during warmup:

$$
\log\sigma_{n+1}=\log\sigma_n+\gamma_n(\alpha_n-\alpha^\star).
$$

Here, \(\alpha_n\) is the acceptance statistic. Cathedral must define it as either the acceptance outcome or the computed MH acceptance probability. It must use one definition consistently. The target rate is \(\alpha^\star\). The value \(\gamma_n\) decreases with the number of visits to the address.

Useful initial targets are about 0.44 for a scalar random walk and 0.234 for a moderate-dimensional block. These values come from asymptotic results under regular target distributions. They are starting points, not universal laws.

The simplest safe rule is:

1. Adapt during warmup.
2. Bound scales away from zero and infinity.
3. Freeze all proposal parameters after warmup.
4. Freeze adapted address-selection probabilities after warmup.
5. Discard warmup samples.

More general continuing adaptation is possible, but it needs diminishing adaptation and containment conditions. Cathedral does not need that complexity for the first implementation.

Cathedral can also adapt how often it selects each address. Output-sensitive lightweight MH selects choices more often when they strongly affect a query. A nonuniform scheduler must include the forward and reverse address-selection probabilities in the MH ratio. Every active address must keep a positive selection probability.

### 4.3 Blocked adaptive Metropolis

Single-address updates move slowly along a correlated posterior ridge. A block proposal updates a vector of choices together:

$$
q'=q+sLz,\qquad z\sim\mathcal{N}(0,I).
$$

The matrix \(L\) is a Cholesky factor of an estimated covariance. Warmup can estimate this covariance online.

The support-transform Jacobian is not the main dense cost. Elementwise transforms give a diagonal Jacobian with an \(O(d)\) log determinant. Dense covariance storage and updates cost \(O(d^2)\). A dense proposal draw costs \(O(d^2)\). A full Cholesky rebuild costs \(O(d^3)\), but it can occur only at selected warmup intervals.

One block with all continuous variables is appropriate when the variables always exist together, have stable shapes, and have an approximately elliptical joint posterior. Smaller blocks are better when:

- only some variables are strongly correlated;
- branches contain mutually exclusive choices;
- covariance estimation is unreliable in the full dimension;
- one difficult direction forces every other variable to use a tiny step.

Cathedral should start with user-declared blocks. It can later cluster variables by correlations estimated during warmup. It can compare candidate partitions by ESS per second. Block moves should remain mixed with single-address moves.

For the external hepatitis benchmark that motivated this review, a block containing all four homogeneous continuous variables is reasonable. A separate discrete kernel should update `k000`. Inside the repository, `examples/linear_regression.py` is a useful first block benchmark.

## 5. Safe local kernels beyond random-walk MH

### 5.1 Scalar and block slice sampling

Slice sampling first draws a density threshold below the current density. It then samples from the set of values above that threshold. This set is the slice. A one-dimensional slice can contain disconnected intervals. Stepping-out or doubling builds a bracket around the current value. Shrinkage preserves the uniform distribution over the bracketed part of the slice.

This method often needs less scale tuning than random-walk MH. It uses a random number of target evaluations. Expensive full replay can therefore reduce its ESS per second.

The basic scalar method should require:

- one continuous choice;
- a deterministic conditional log density when other choices are fixed;
- unchanged trace structure across candidate values;
- a valid bounded or transformed domain.

Multivariate slice methods also exist. Random-direction and hit-and-run methods update a block along a line. Crumb methods use rejected trial points to learn a local proposal region. Elliptical slice sampling is especially useful for a fixed block with a multivariate Normal prior. It needs no step-size parameter.

Slice sampling is not a general structural kernel. If each density evaluation creates fresh descendants, the density estimate becomes random and ordinary slice correctness no longer applies.

### 5.2 Automatic finite-support Gibbs

Cathedral can implement a useful generic Gibbs kernel without symbolic conjugacy.

For a choice with finite support:

1. Replay the model once for every possible value.
2. Hold every other semantic choice fixed.
3. Record the complete unnormalized target, represented by `trace.log_joint` under the future validated semantics.
4. Normalize the candidate scores.
5. Draw from the exact conditional category.

This kernel is exact when changing the choice preserves the remaining trace structure. It costs one replay per candidate. A finite block costs the product of its support sizes.

Changes in trace structure are the main limit. If a candidate creates descendants, its Gibbs weight must integrate over all values of those descendants. Cathedral can sometimes use exact enumeration when all new descendants have finite support.

Drawing one descendant set for each candidate is not exact Gibbs. It is an auxiliary-variable proposal and requires an MH correction.

Automatic finite Gibbs is therefore appropriate for Booleans, small categorical variables, and bounded count variables in stable traces. Poisson variables need another kernel. Finite truncation changes the target unless Cathedral represents or corrects the omitted tail mass exactly. Metropolized Gibbs can reduce self-transitions after Cathedral has calculated all candidate scores.

### 5.3 Locally informed discrete proposals

Large discrete spaces make full Gibbs too expensive. Locally balanced proposals score a neighborhood and prefer promising nearby states. An MH correction preserves the target.

Examples include:

- adjacent integer moves;
- category changes weighted by conditional scores;
- partition moves;
- local tree or grammar edits;
- branch changes among a finite set of alternatives.

This method is important for Cathedral because many of its distinctive models are discrete or mixed. Proposal metadata should let a choice define its neighborhood separately from its finite support.

### 5.4 Delayed rejection and multiple-try Metropolis

Delayed rejection makes a second, usually smaller, proposal after the first proposal fails. The second acceptance rule must include the probability that the first proposal was rejected in both directions. Simply retrying ordinary MH with a smaller scale is not correct.

Multiple-try Metropolis creates several candidates, selects one by weighted probabilities, and constructs a reverse reference set. It can use parallel model evaluations. It also spends roughly \(2K-1\) target evaluations for \(K\) forward candidates in its original form.

Delayed rejection is the better early refinement. Multiple-try Metropolis should wait until Cathedral can evaluate complete traces in parallel processes.

## 6. Population and global methods

### 6.1 Annealed Sequential Monte Carlo over complete traces

Annealed SMC is the recommended first global method. It defines intermediate targets:

$$
\pi_\beta(\tau)\propto p_{\mathrm{prior}}(\tau)C(\tau)\exp\!\left(\beta\ell(\tau)\right),\qquad 0=\beta_0<\cdots<\beta_T=1.
$$

At \(\beta=0\), the target is the prior restricted to traces that satisfy all hard conditions. It is the unconditional prior only if all prior traces satisfy those conditions. At \(\beta=1\), the target is the posterior. Intermediate values of \(\beta\) gradually introduce finite scores from `observe()` and `factor()`.

A complete-trace particle contains:

```text
Trace
+ log weight
+ temperature
+ ancestor identifier
+ RNG stream identifier
+ cached prior and score
+ optional mutation-kernel state
```

One stage performs:

```text
increase temperature
→ update weights
→ measure particle ESS
→ resample if required
→ mutate each trace with valid kernels
```

Cathedral stores choice log probabilities separately from `log_score`. It can therefore temper finite scores without tempering prior choice probabilities. Different particles can have different address sets and dimensions.

After resampling, copied particles must receive independently split child RNG streams. Copying one RNG state into all children can make the mutations identical.

Annealed SMC gives several benefits:

- population exploration instead of one chain;
- support for continuous, discrete, mixed, and changing traces;
- a normalizing-constant estimate;
- natural use of all local kernels as particle mutation kernels;
- useful parallel structure when the runtime is ready.

It also has limits.

- Poor mutation kernels produce many duplicate particles.
- Global continuous parameters can still cause particle collapse.
- Rare hard `condition()` calls cannot be gradually softened automatically.
- Adaptive temperature schedules need careful normalizing-constant tests.
- Memory grows with particle count and trace size.

Rare hard conditions require constrained initialization or a user-defined bridge. If annealing starts from a constrained prior, Cathedral must account for its unknown normalizing constant. Cathedral cannot otherwise report the final estimate as model evidence.

An arbitrary `factor()` contributes to a normalizing constant. That constant is a Bayesian marginal likelihood only when the factor represents a normalized observation model.

Cathedral should select temperature increments by conditional particle ESS. It should report particle ESS, unique ancestors, resampling count, model executions, and normalizing-constant estimates.

### 6.2 Parallel tempering

Parallel tempering runs replicas at fixed inverse temperatures. Each replica applies a valid local kernel for its target. A swap kernel occasionally exchanges two complete traces.

Hot replicas move through broad regions. The cold replica at \(\beta=1\) targets the posterior. Complete-trace swaps need no dimension-matching Jacobian because the method exchanges entire states.

Parallel tempering is compatible with Cathedral's mixed trace space. It is not automatically the next priority.

- It multiplies local-kernel cost by the number of replicas.
- It does not repair a bad local proposal.
- A poor temperature ladder produces few useful round trips.
- Annealed SMC gives posterior particles and a normalizing-constant estimate in one population run.

Cathedral should add parallel tempering only after adapted local kernels and annealed SMC. A representative multimodal benchmark should show a clear benefit in mode occupancy and effective samples per second.

### 6.3 Annealed Importance Sampling and tempered transitions

Annealed Importance Sampling (AIS) runs independent weighted trajectories through an annealing schedule without resampling. AIS is useful for normalizing-constant estimation. It can reuse the target and kernel interfaces from annealed SMC.

Tempered transitions use one chain. A proposal travels up a temperature ladder and back down before one final MH decision. This method can create large global moves without maintaining permanent replicas.

These methods share infrastructure with annealed SMC and parallel tempering. They should not each create a separate scoring implementation.

## 7. Efficient inference for changing trace structure

### 7.1 Keep lightweight trace MH as the universal fallback

Cathedral's current prior-proposal MH kernel remains a useful fallback. A prior proposal can create or remove downstream choices. The MH kernel's structural corrections require stable addresses and a replay that records every random decision. The guarantee does not cover `DPmem()` allocation decisions, hidden randomness, duplicate names, or address aliasing.

The prior-proposal kernel should remain available after Cathedral adds local proposals. A mixture of local proposals and prior proposals can improve movement within one structure. The prior-proposal component keeps access to other structures.

### 7.2 Programmable multi-choice proposals

The next structural step should be an ordinary proposal-program API, not a complete reversible-jump framework.

A proposal program can:

- inspect the current trace;
- select a related set of addresses;
- sample auxiliary values;
- return coordinated interventions;
- report forward and reverse proposal scores.
- report a log absolute Jacobian determinant when a continuous deterministic map requires one.

The proposal-program API supports branch changes with compatible descendant values. Early proposal programs should use discrete changes or independently sampled births and deaths. Continuous split-merge transformations belong in the later reversible-jump layer. They require explicit dimension matching and Jacobian support.

Gen demonstrates the value of making proposal programs first-class. Trace types and guide types show that support matching is a central correctness obligation. Cathedral can begin with runtime validation before it considers a static type system.

### 7.3 Reversible-jump and involutive MCMC

Reversible-jump MCMC formalizes moves between spaces with different dimensions. A move uses auxiliary randomness and a dimension-matching bijection:

$$
(\tau,u) \longleftrightarrow (\tau',u').
$$

The acceptance ratio includes target scores, forward and reverse auxiliary densities, move-type probabilities, and the absolute Jacobian determinant of the continuous map.

Involutive MCMC expresses this transformation as a self-inverse operation on an extended state. The involutive framework includes MH, reversible jump, and many auxiliary-variable kernels.

These methods fit Cathedral's unique goals. They do not require a gradient of the complete target. They do require strong infrastructure:

- stable hierarchical addresses;
- explicit auxiliary proposal traces;
- trace differences;
- forward and reverse assessment;
- Jacobian support for continuous proposal maps;
- round-trip and support validation.

Cathedral should first implement targeted structural moves for real models. A generic involutive API should follow only after those moves reveal the required abstractions.

### 7.4 Path decomposition and stochastic support

Another strategy divides execution into paths with fixed support. Cathedral can use a suitable local method within each path. A combining step weights each path estimate by its estimated posterior probability or normalizing constant.

Path decomposition does not require one trans-dimensional chain to cross rare control-flow boundaries. It can also reveal unstable path weights and weakly identified structures. Recent work studies path decomposition and Bayesian model averaging across program paths.

Path-based methods can complement, rather than replace, structural MCMC and SMC.

### 7.5 Incremental execution

Every current MH step reruns the complete Python model. Incremental execution can reduce that cost.

Relevant techniques include:

- program slicing;
- dependency-aware trace regeneration;
- continuations;
- call-site caching;
- conservative trace update;
- aligned checkpoints;
- static factorization.

Incremental execution reduces the cost of a transition. It does not improve the transition's statistical quality.

Arbitrary Python makes safe incrementality difficult. File I/O, mutation, object identity, exceptions, and hidden dependencies can invalidate a cache. Cathedral should first build a limited architecture prototype with explicit pure or cacheable model regions. Full replay must remain the correctness fallback.

### 7.6 Sequential SMC and particle MCMC

Complete-trace annealed SMC operates on finished traces. Sequential SMC pauses particles at `observe()` calls or explicit resampling checkpoints. Sequential SMC then reweights and copies partial executions.

A sequential particle needs:

- a partial trace;
- a suspended continuation or replay checkpoint;
- local variables and relevant heap state;
- memoization state;
- address counters and control-flow history;
- incremental weight;
- ancestor information;
- RNG state.

An ordinary Python call stack cannot be cloned. Cathedral would need source transformation, continuation-passing style, replay from checkpoints, process-level copy-on-write, or a restricted `SequentialModel` API.

Particle MCMC places an SMC algorithm inside an MCMC transition. Particle independent MH proposes a complete trajectory with SMC. Particle Gibbs conditions the particle system on one reference trajectory from the previous MCMC iteration. Ancestor sampling reduces path degeneracy.

These methods are transformative for state-space and sequential models. They should remain a separate development branch until such workloads become strategic.

### 7.7 Delayed sampling, inference combinators, and learned proposals

Delayed sampling detects conjugate relationships at runtime. It postpones a draw and updates the symbolic conditional distribution when observations arrive. It samples only when the value is required. Delayed sampling can provide automatic Rao-Blackwellization and locally optimal SMC proposals.

Inference combinators give proper-weighting guarantees to proposal, extension, resampling, and composition operations. SMCP3 adds forward and backward proposal programs, auxiliary randomness, and deterministic transformations. Both frameworks are relevant to Cathedral's long-term kernel system.

Learned proposals can use observations to suggest values for dynamic addresses. The final sampler can remain exact when MH or importance corrections use a tractable proposal density with adequate support. Training cost makes this most useful for repeated inference across related datasets.

Recent systems also vectorize programmable inference over particles and chains. They preserve stochastic branching. Cathedral should study these systems after it has stable proposal and particle representations.

## 8. A staged Cathedral roadmap

### Phase 0: semantic and diagnostic hardening

**Purpose:** Make advanced inference failures visible before adding more algorithms.

Minimum deliverables:

- reject duplicate explicit addresses;
- require explicit stable names for initial advanced kernels;
- design hierarchical or scoped address identity for loops, recursion, and memoized calls;
- document a replay and purity contract;
- separate score events or at least classify temperable scores and hard constraints;
- add domain, dtype, event-shape, bound, and finite-support metadata;
- accept user-supplied valid initial traces and add a constraint-guided initialization track;
- review `DPmem()` so allocation randomness is represented in replayable trace state.

**Benefit:** Every later kernel can make valid eligibility decisions.

The address, replay, score, and support items gate Phase 1. Cathedral can develop resumable sampler state and multi-chain diagnostics in parallel. These features must be available before Cathedral releases adaptation. Dynamic diagnostics must include path and address-presence indicators.

**Success gate:** Tiny finite models agree with both exact enumeration and hand-derived answers. Hand-derived transition matrices have small detailed-balance residuals. Structural proposal round trips recover the original trace and auxiliary values. Invalid address reuse fails immediately.

### Phase 1: general proposal and kernel interfaces

**Purpose:** Replace the hard-coded prior proposal without changing the default MH behavior.

Deliverables:

- proposal results with forward and reverse log probabilities;
- complete forward and reverse proposal densities after support transformations;
- an address-selector interface;
- `Kernel`, `CompoundKernel`, mixture, cycle, and repeat combinators;
- explicit adaptation state and diagnostics;
- a prior-proposal implementation that reproduces current results.

**Benefit:** All later methods share one composition and correctness layer.

**Success gate:** Existing MH tests remain unchanged. New detailed-balance tests use analytic targets and structural round-trip checks in addition to Cathedral's enumeration engine.

### Phase 2: support-aware adaptive local MH

**Purpose:** Fix the largest immediate efficiency problem.

Deliverables:

- transformed random walks for real, positive, bounded, integer, and simplex choices, with correct proposal densities or target Jacobians;
- local-plus-prior proposal mixtures;
- per-address warmup-only scale adaptation;
- per-address acceptance and jump-distance reports.

**Benefit:** Concentrated posteriors no longer depend on lucky prior redraws. Dynamic branch discovery remains possible.

**Success gate:** Pre-register model-execution and wall-time budgets. Report estimates and confidence intervals across seeds. Continue development only if the method gives a material improvement without biased posterior estimates. A fivefold ESS-per-second improvement on one concentrated benchmark is a useful target.

### Phase 3: finite Gibbs and scalar slice

**Purpose:** Add safe kernels that remove avoidable rejection.

Deliverables:

- exact finite-choice Gibbs for structure-preserving choices;
- optional Metropolized Gibbs;
- scalar stepping-out and shrinkage slice kernels;
- explicit eligibility checks and fallback behavior.

**Benefit:** Better discrete inference and a tuning-light continuous fallback.

**Success gate:** Gibbs matches enumeration. Slice matches analytic scalar targets. Structure-changing candidates fail clearly instead of silently using an invalid kernel.

### Phase 4: user-defined adaptive blocks

**Purpose:** Address linear posterior correlation.

Deliverables:

- named fixed-shape block specifications;
- packing and transforms for mixed continuous support types;
- regularized online covariance adaptation;
- block and single-address kernel mixtures;
- later, optional elliptical slice for Gaussian-prior blocks.

**Benefit:** Large improvement for regression, hierarchical parameters, and low-dimensional correlated models.

**Success gate:** Compare against scalar updates under pre-registered budgets and replicated seeds. Continue development only if blocks materially increase ESS per second for a highly correlated Gaussian and Cathedral's regression example. Covariance estimates must remain positive-definite.

### Phase 5: complete-trace annealed SMC

**Purpose:** Add a general population method for mixed and variable structures.

Deliverables:

- particle state and ancestry;
- fixed temperature schedules and schedules that adapt to particle ESS;
- stable log-weight calculations;
- resampling strategies;
- kernel-based particle mutation;
- normalizing-constant and particle diagnostics.

**Benefit:** Better global exploration, posterior particles, and normalizing-constant estimation. This directly extends Cathedral's current importance sampler.

**Success gate:**

- Dynamic finite models match analytic answers and enumeration.
- Repeated runs recover known normalizing constants.
- Cathedral uses the term _Bayesian evidence_ only when the normalization and score semantics justify it.
- Under a fixed model-execution budget, annealed SMC gives lower estimation error than prior importance sampling on concentrated models.

### Phase 6: targeted programmable structural proposals

**Purpose:** Make deliberate moves between trace structures.

Deliverables:

- proposal programs with auxiliary randomness;
- compute trace differences and forward and reverse proposal scores;
- discrete or independently sampled birth-death examples;
- explicit Jacobian support before continuous split-merge examples;
- runtime support and round-trip checks.

**Benefit:** Cathedral becomes efficient on the models that justify its Church-style expressiveness.

**Success gate:** Structural toy models match exact model probabilities. Proposal and inverse round trips recover the original values and auxiliaries. Jacobians pass finite-difference checks when present. The first real example should be a small mixture, tree, grammar, or one-versus-two-group model.

### Demand-driven branches

After Phase 6, use benchmark results and user workloads to select more work.

**Parallel tempering:** Add when adapted local kernels and SMC still miss persistent modes.

**Sequential SMC and particle MCMC:** Add when time-series or state-space models become a primary use case. Prefer a restricted sequential API over an immediate attempt to clone arbitrary Python continuations.

**Incremental execution:** Add when profiling shows that full model replay, rather than poor mixing, dominates cost. Keep full replay as a fallback.

**Generic involutive MCMC:** Add after several targeted structural moves establish stable proposal and Jacobian abstractions.

**Learned proposals:** Add for repeated inference workloads after Cathedral has batching, stable observation schemas, and exact proposal-density accounting.

## 9. Method selection by model type

No new method should replace an existing method when the existing method is already best.

- **Small finite models:** Use exact enumeration.
- **Easy hard conditions:** Use rejection sampling.
- **Broad priors with informative observations:** Use adaptive local MH or annealed SMC.
- **Low-dimensional correlated continuous models:** Use adaptive blocks.
- **Stable scalar continuous choices:** Use slice or adaptive local MH.
- **Small finite discrete choices:** Use exact Gibbs.
- **Large local discrete spaces:** Use locally balanced MH.
- **Mixed stable models:** Use a compound cycle of block MH, scalar kernels, and discrete Gibbs.
- **Multimodal mixed models:** Use annealed SMC first. Add parallel tempering if benchmarks justify it.
- **Changing trace structures:** Keep prior-resimulation MH. Add targeted structural proposal programs.
- **Sequential state-space models:** Use sequential SMC and particle MCMC after a checkpointable execution API exists.
- **Repeated related inference tasks:** Consider learned proposals with exact corrections.

For the external hepatitis model that motivated this review, the near-term non-NUTS design is:

1. Put the continuous variables in one transformed adaptive block.
2. Update `k000` with a local integer proposal or a justified conditional calculation.
3. Cycle both kernels through a compound interface.
4. Run multiple chains and report ESS, MCSE, and \(\widehat{R}\).
5. Compare the result with complete-trace annealed SMC.

This model is small enough that these methods can provide most of the practical benefit without a differentiable NUTS backend.

## 10. Conclusion

Cathedral's path to efficient inference should follow its model semantics.

Cathedral is strongest when a model is an ordinary Python program. Such a program can use stochastic control flow and a changing set of choices. A fixed-dimensional gradient sampler cannot organize all inference for this system.

A shared proposal and kernel system is the foundation. Support-aware adaptation makes local MH more efficient. Gibbs, slice, and block kernels handle eligible subproblems. Annealed SMC supports global exploration across mixed trace spaces. Programmable proposals handle deliberate changes in trace structure.

With that foundation, Cathedral can add sequential particle methods, involutive MCMC, incremental execution, and learned proposals. They become integrated extensions instead of isolated algorithms.

This roadmap does not reject NUTS. It defines an inference strategy that matches Cathedral's trace-based execution model.

## References

### A. General reviews and books

- **Andrieu, C., de Freitas, N., Doucet, A., and Jordan, M. I. (2003).** [An Introduction to MCMC for Machine Learning](https://doi.org/10.1023/A:1020281327116). _Machine Learning_ 50. A concise review of MH, Gibbs, blocking, and auxiliary variables.
- **Robert, C. P., and Casella, G. (2004).** [Monte Carlo Statistical Methods](https://doi.org/10.1007/978-1-4757-4145-2), second edition. Springer. A broad statistical reference.
- **Craiu, R. V., Vats, D., Jones, G. L., Brooks, S., Gelman, A., and Meng, X.-L., editors (2026).** [Handbook of Markov Chain Monte Carlo](https://doi.org/10.1201/9781003453420), second edition. CRC Press. A current reference on MCMC theory, diagnostics, multimodality, and implementation.
- **Chopin, N., and Papaspiliopoulos, O. (2020).** [An Introduction to Sequential Monte Carlo](https://doi.org/10.1007/978-3-030-47845-2). Springer. A modern SMC textbook.
- **Naesseth, C. A., Lindsten, F., and Schön, T. B. (2019).** [Elements of Sequential Monte Carlo](https://doi.org/10.1561/2200000074). _Foundations and Trends in Machine Learning_ 12(3). A tutorial on SMC targets, proposals, weights, and learning.
- **van de Meent, J.-W., Paige, B., Yang, H., and Wood, F. (2018).** [An Introduction to Probabilistic Programming](https://arxiv.org/abs/1809.10756). A book-length review of traces, MH, SMC, and dynamic models.
- **Hastings, W. K. (1970).** [Monte Carlo Sampling Methods Using Markov Chains and Their Applications](https://doi.org/10.1093/biomet/57.1.97). _Biometrika_ 57(1). The general asymmetric MH rule.
- **Tierney, L. (1994).** [Markov Chains for Exploring Posterior Distributions](https://doi.org/10.1214/aos/1176325750). _Annals of Statistics_ 22(4). General-state-space MCMC foundations.
- **Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., and Bürkner, P.-C. (2021).** [Rank-Normalization, Folding, and Localization: An Improved \(\widehat{R}\) for Assessing Convergence of MCMC](https://doi.org/10.1214/20-BA1221). _Bayesian Analysis_ 16(2). Modern \(\widehat{R}\), ESS, and MCSE diagnostics.

### B. Universal probabilistic programming and trace MH

- **Goodman, N. D., Mansinghka, V. K., Roy, D. M., Bonawitz, K., and Tenenbaum, J. B. (2008).** [Church: A Language for Generative Models](https://proceedings.mlr.press/r6/goodman08a.html). _UAI 2008_. The main language ancestor of Cathedral.
- **Wingate, D., Stuhlmüller, A., and Goodman, N. D. (2011).** [Lightweight Implementations of Probabilistic Programming Languages via Transformational Compilation](https://proceedings.mlr.press/v15/wingate11a.html). _AISTATS 2011_. The named-choice trace-MH pattern.
- **Goodman, N. D., and Stuhlmüller, A. (2014).** [The Design and Implementation of Probabilistic Programming Languages](https://dippl.org/). Electronic book. The main implementation reference for WebPPL, including MCMC, enumeration, and particle filtering.
- **Mansinghka, V. K., Selsam, D., and Perov, Y. (2014).** [Venture: A Higher-Order Probabilistic Programming Platform with Programmable Inference](https://arxiv.org/abs/1404.0099). Trace scaffolds and stochastic regeneration.
- **Hur, C.-K., Nori, A. V., Rajamani, S. K., and Samuel, S. (2015).** [A Provably Correct Sampler for Probabilistic Programs](https://doi.org/10.4230/LIPIcs.FSTTCS.2015.475). _FSTTCS 2015_. Correct trace reuse under branches and loops.
- **Borgström, J., Dal Lago, U., Gordon, A. D., and Szymczak, M. (2016).** [A Lambda-Calculus Foundation for Universal Probabilistic Programming](https://doi.org/10.1145/2951913.2951942). _ICFP 2016_. Trace semantics and convergence.
- **Tolpin, D., van de Meent, J.-W., Yang, H., and Wood, F. (2017).** [Design and Implementation of Probabilistic Programming Language Anglican](https://doi.org/10.1145/3064899.3064910). _IFL 2016 proceedings_. CPS execution and functional inference representations.
- **Mansinghka, V. K., et al. (2018).** [Probabilistic Programming with Programmable Inference](https://doi.org/10.1145/3192366.3192409). _PLDI 2018_. Inference metaprogramming and trace subproblems.
- **Cusumano-Towner, M. F., Saad, F. A., Lew, A. K., and Mansinghka, V. K. (2019).** [Gen: A General-Purpose Probabilistic Programming System with Programmable Inference](https://doi.org/10.1145/3314221.3314642). _PLDI 2019_. Proposal programs and trace update operations.
- **Lew, A. K., et al. (2020).** [Trace Types and Denotational Semantics for Sound Programmable Inference](https://doi.org/10.1145/3371087). _POPL 2020_. Static support and soundness obligations.

### C. Adaptive, blocked, and discrete MCMC

- **Gelman, A., Roberts, G. O., and Gilks, W. R. (1996).** [Efficient Metropolis Jumping Rules](https://doi.org/10.1093/oso/9780198523567.003.0038). In _Bayesian Statistics 5_. The source of the approximate 0.44 scalar random-walk heuristic.
- **Roberts, G. O., Gelman, A., and Gilks, W. R. (1997).** [Weak Convergence and Optimal Scaling of Random Walk Metropolis Algorithms](https://doi.org/10.1214/aoap/1034625254). _Annals of Applied Probability_ 7(1). The source of the 0.234 high-dimensional scaling heuristic.
- **Haario, H., Saksman, E., and Tamminen, J. (2001).** [An Adaptive Metropolis Algorithm](https://doi.org/10.2307/3318737). _Bernoulli_ 7(2). Empirical-covariance adaptation.
- **Roberts, G. O., and Rosenthal, J. S. (2007).** [Coupling and Ergodicity of Adaptive MCMC Algorithms](https://doi.org/10.1239/jap/1183667414). _Journal of Applied Probability_ 44(2). Diminishing adaptation and containment.
- **Andrieu, C., and Thoms, J. (2008).** [A Tutorial on Adaptive MCMC](https://doi.org/10.1007/s11222-008-9110-y). _Statistics and Computing_ 18. Practical adaptation methods and correctness requirements.
- **Vihola, M. (2012).** [Robust Adaptive Metropolis Algorithm with Coerced Acceptance Rate](https://doi.org/10.1007/s11222-011-9269-5). _Statistics and Computing_ 22. Stable scale and shape adaptation.
- **Tolpin, D., van de Meent, J.-W., Paige, B., and Wood, F. (2015).** [Output-Sensitive Adaptive Metropolis-Hastings for Probabilistic Programs](https://doi.org/10.1007/978-3-319-23525-7_19). _ECML PKDD 2015_. Adaptive site scheduling in trace MH.
- **Turek, D., de Valpine, P., Paciorek, C. J., and Anderson-Bergman, C. (2017).** [Automated Parameter Blocking for Efficient Markov Chain Monte Carlo Sampling](https://doi.org/10.1214/16-BA1008). _Bayesian Analysis_ 12(2). Correlation-based block construction.
- **Liu, J. S. (1994).** [The Collapsed Gibbs Sampler in Bayesian Computations with Applications to a Gene Regulation Problem](https://doi.org/10.1080/01621459.1994.10476829). _JASA_ 89(427). Marginalization and dependence reduction.
- **Liu, J. S. (1996).** [Metropolized Gibbs Sampler: An Improvement](https://doi.org/10.1093/biomet/83.3.681). _Biometrika_ 83(3). A finite-conditional update that avoids unnecessary self-transitions.
- **Roberts, G. O., and Sahu, S. K. (1997).** [Updating Schemes, Correlation Structure, Blocking and Parameterization for the Gibbs Sampler](https://doi.org/10.1111/1467-9868.00070). _JRSS B_ 59(2). The effect of schedules and blocks.
- **Liu, J. S., Liang, F., and Wong, W. H. (2000).** [The Multiple-Try Method and Local Optimization in Metropolis Sampling](https://doi.org/10.1080/01621459.2000.10473908). _JASA_ 95(449). The original multiple-try Metropolis construction.
- **Green, P. J., and Mira, A. (2001).** [Delayed Rejection in Reversible Jump Metropolis-Hastings](https://doi.org/10.1093/biomet/88.4.1035). _Biometrika_ 88(4). Correct second-stage proposals, including variable-dimensional moves.
- **Zanella, G. (2020).** [Informed Proposals for Local MCMC in Discrete Spaces](https://doi.org/10.1080/01621459.2019.1585255). _JASA_ 115(530). Locally balanced discrete proposals.

### D. Slice sampling

- **Neal, R. M. (2003).** [Slice Sampling](https://doi.org/10.1214/aos/1056562461). _Annals of Statistics_ 31(3). Scalar and multivariate slice foundations.
- **Walker, S. G. (2007).** [Sampling the Dirichlet Mixture Model with Slices](https://doi.org/10.1080/03610910601096262). Infinite-mixture augmentation by slices.
- **Murray, I., Adams, R., and MacKay, D. J. C. (2010).** [Elliptical Slice Sampling](https://proceedings.mlr.press/v9/murray10a.html). _AISTATS 2010_. A tuning-free Gaussian-prior block kernel.
- **Ranca, R., and Ghahramani, Z. (2015).** [Slice Sampling for Probabilistic Programming](https://arxiv.org/abs/1501.04684). A trans-dimensional trace-slice proposal; non-archival preprint.

### E. Tempering, AIS, and SMC

- **Marinari, E., and Parisi, G. (1992).** [Simulated Tempering](https://doi.org/10.1209/0295-5075/19/6/002). Temperature as an auxiliary state.
- **Hukushima, K., and Nemoto, K. (1996).** [Exchange Monte Carlo Method](https://doi.org/10.1143/JPSJ.65.1604). Replica exchange and parallel tempering.
- **Earl, D. J., and Deem, M. W. (2005).** [Parallel Tempering: Theory, Applications, and New Perspectives](https://doi.org/10.1039/B509983H). _Physical Chemistry Chemical Physics_ 7. A broad review of replica exchange design and applications.
- **Neal, R. M. (1996).** [Sampling from Multimodal Distributions Using Tempered Transitions](https://doi.org/10.1007/BF00143556). _Statistics and Computing_ 6. Temperature round-trip proposals.
- **Neal, R. M. (2001).** [Annealed Importance Sampling](https://doi.org/10.1023/A:1008923215028). _Statistics and Computing_ 11. Weighted annealing paths and evidence.
- **Del Moral, P., Doucet, A., and Jasra, A. (2006).** [Sequential Monte Carlo Samplers](https://doi.org/10.1111/j.1467-9868.2006.00553.x). _JRSS B_ 68(3). General SMC samplers with forward and backward kernels.
- **Andrieu, C., Doucet, A., and Holenstein, R. (2010).** [Particle Markov Chain Monte Carlo Methods](https://doi.org/10.1111/j.1467-9868.2009.00736.x). _JRSS B_ 72(3). PMMH and particle Gibbs.
- **Zhou, Y., Johansen, A. M., and Aston, J. A. D. (2016).** [Toward Automatic Model Comparison: An Adaptive Sequential Monte Carlo Approach](https://doi.org/10.1080/10618600.2015.1060885). _JCGS_ 25(3). Conditional ESS and adaptive annealing schedules.

### F. SMC and particle methods in probabilistic programs

- **Wood, F., van de Meent, J.-W., and Mansinghka, V. K. (2014).** [A New Approach to Probabilistic Programming Inference](https://proceedings.mlr.press/v33/wood14.html). _AISTATS 2014_. Particle Gibbs for expressive programs.
- **van de Meent, J.-W., Yang, H., Mansinghka, V. K., and Wood, F. (2015).** [Particle Gibbs with Ancestor Sampling for Probabilistic Programs](https://proceedings.mlr.press/v38/vandemeent15.html). _AISTATS 2015_. Reduced path degeneracy.
- **Rainforth, T., et al. (2016).** [Interacting Particle Markov Chain Monte Carlo](https://proceedings.mlr.press/v48/rainforth16.html). _ICML 2016_. Parallel SMC and conditional SMC nodes.
- **Murray, L. M., et al. (2018).** [Delayed Sampling and Automatic Rao-Blackwellization of Probabilistic Programs](https://proceedings.mlr.press/v84/murray18a.html). _AISTATS 2018_. Runtime conjugacy and locally optimal proposals.
- **Lundén, D., Borgström, J., and Broman, D. (2021).** [Correctness of Sequential Monte Carlo Inference for Probabilistic Programming Languages](https://doi.org/10.1007/978-3-030-72019-3_15). _ESOP 2021_. Correct resampling checkpoints.
- **Lew, A. K., et al. (2023).** [SMCP3: Sequential Monte Carlo with Probabilistic Program Proposals](https://proceedings.mlr.press/v206/lew23a.html). _AISTATS 2023_. Forward and backward proposal programs with automated weights.

### G. Structural, reversible-jump, and involutive MCMC

- **Green, P. J. (1995).** [Reversible Jump Markov Chain Monte Carlo Computation and Bayesian Model Determination](https://doi.org/10.1093/biomet/82.4.711). _Biometrika_ 82(4). The trans-dimensional foundation.
- **Richardson, S., and Green, P. J. (1997).** [Bayesian Analysis of Mixtures with an Unknown Number of Components](https://doi.org/10.1111/1467-9868.00095). _JRSS B_ 59(4). Birth, death, split, and merge moves.
- **Jain, S., and Neal, R. M. (2004).** [A Split-Merge MCMC Procedure for the Dirichlet Process Mixture Model](https://doi.org/10.1198/1061860043001). _JCGS_ 13(1). Large cluster-structure moves.
- **Roberts, D. A., Gallagher, M., and Taimre, T. (2019).** [Reversible Jump Probabilistic Programming](https://proceedings.mlr.press/v89/roberts19a.html). _AISTATS 2019_. Automatic RJ calculations from target and proposal programs.
- **Zhou, Y., Yang, H., Teh, Y. W., and Rainforth, T. (2020).** [Divide, Conquer, and Combine](https://proceedings.mlr.press/v119/zhou20e.html). _ICML 2020_. Fixed-support path subproblems for stochastic-support programs.
- **Neklyudov, K., Welling, M., Egorov, E., and Vetrov, D. (2020).** [Involutive MCMC: A Unifying Framework](https://proceedings.mlr.press/v119/neklyudov20a.html). _ICML 2020_. A common framework for MH and reversible jump.
- **Cusumano-Towner, M. F., Lew, A. K., and Mansinghka, V. K. (2020).** [Automating Involutive MCMC Using Probabilistic and Differentiable Programming](https://arxiv.org/abs/2007.09871). Automated trace and Jacobian accounting in Gen.
- **Mak, C., Zaiser, F., and Ong, C.-H. L. (2022).** [Nonparametric Involutive Markov Chain Monte Carlo](https://proceedings.mlr.press/v162/mak22a.html). _ICML 2022_. Involutive kernels for unbounded program traces.
- **Reichelt, T., Ong, C.-H. L., and Rainforth, T. (2024).** [Beyond Bayesian Model Averaging over Paths in Probabilistic Programs with Stochastic Support](https://proceedings.mlr.press/v238/reichelt24a.html). _AISTATS 2024_. Path weights and stochastic-support interpretation.

### H. Programmable, learned, and sound proposal systems

- **Stuhlmüller, A., Taylor, J., and Goodman, N. D. (2013).** [Learning Stochastic Inverses](https://proceedings.neurips.cc/paper/2013/hash/7f53f8c6c730af6aeb52e66eb74d8507-Abstract.html). _NeurIPS 2013_. Learned inverse-conditionals for blocked proposals.
- **Le, T. A., Baydin, A. G., and Wood, F. (2017).** [Inference Compilation and Universal Probabilistic Programming](https://proceedings.mlr.press/v54/le17a.html). _AISTATS 2017_. Learned address-wise trace proposals.
- **Wang, T., Wu, Y., Moore, D. A., and Russell, S. (2018).** [Meta-Learning MCMC Proposals](https://proceedings.neurips.cc/paper/2018/hash/584b98aac2dddf59ee2cf19ca4ccb75e-Abstract.html). _NeurIPS 2018_. Reusable learned block-Gibbs kernels.
- **Liang, F., et al. (2021).** [Accelerating Metropolis-Hastings with Lightweight Inference Compilation](https://proceedings.mlr.press/v130/liang21a.html). _AISTATS 2021_. Learned corrected proposals for dynamic Markov blankets.
- **Stites, S., Zimmermann, H., Wu, H., Sennesh, E., and van de Meent, J.-W. (2021).** [Learning Proposals for Probabilistic Programs with Inference Combinators](https://proceedings.mlr.press/v161/stites21a.html). _UAI 2021_. Properly weighted proposal, resample, compose, and extend operations.
- **Pham, L., Wang, D., Saad, F. A., and Hoffmann, J. (2024).** [Programmable MCMC with Soundly Composed Guide Programs](https://doi.org/10.1145/3689748). _OOPSLA 2024_. Sound composition of recursive block-MH guides.

### I. Incremental execution, alignment, and emerging systems

- **Yang, L., Hanrahan, P., and Goodman, N. D. (2014).** [Generating Efficient MCMC Kernels from Probabilistic Programs](https://proceedings.mlr.press/v33/yang14d.html). _AISTATS 2014_. Program slicing for selective recomputation.
- **Ritchie, D., Stuhlmüller, A., and Goodman, N. D. (2016).** [C3: Lightweight Incrementalized MCMC](https://proceedings.mlr.press/v51/ritchie16.html). _AISTATS 2016_. Continuations and call-site caching.
- **Cusumano-Towner, M., et al. (2018).** [Incremental Inference for Probabilistic Programs](https://doi.org/10.1145/3192366.3192399). _PLDI 2018_. SMC trace translation between related programs, not same-execution replay caching.
- **Lundén, D., et al. (2023).** [Automatic Alignment in Higher-Order Probabilistic Programming Languages](https://doi.org/10.1007/978-3-031-30044-8_20). _ESOP 2023_. Shared checkpoints for SMC and lightweight MH.
- **Becker, M. R., et al. (2026).** [Probabilistic Programming with Vectorized Programmable Inference](https://doi.org/10.1145/3776729). _POPL 2026_. Vectorized traces, particles, chains, and stochastic branching.
- **Böck, M., and Cito, J. (2026).** [Static Factorisation of Probabilistic Programs with User-Labelled Sample Statements and While Loops](https://doi.org/10.1145/3798223). _OOPSLA 2026_. Static dependence factorization for dynamic-address programs.
