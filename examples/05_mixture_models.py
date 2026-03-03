"""Mixture Models and Non-parametrics -- Cathedral examples.

Demonstrates mixture models, unknown number of clusters,
and the Dirichlet Process via DPmem.

Church (v1): https://v1.probmods.org/mixture-models.html
WebPPL (v2): https://probmods.org/chapters/mixture-models.html
"""

from cathedral import (
    Categorical,
    DPmem,
    Normal,
    UniformDraw,
    factor,
    flip,
    infer,
    mem,
    model,
    observe,
    sample,
)

# ---------------------------------------------------------------------------
# 1. Known mixture: two Gaussians
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. Two-component Gaussian mixture (known assignment)")
print("=" * 60)


@model
def known_mixture(data):
    """Two-component Gaussian mixture with equal mixing weights."""
    mu1 = sample(Normal(0, 5), name="mu1")
    mu2 = sample(Normal(0, 5), name="mu2")

    for x in data:
        if flip(0.5):
            observe(Normal(mu1, 0.5), x)
        else:
            observe(Normal(mu2, 0.5), x)

    return {"mu1": min(mu1, mu2), "mu2": max(mu1, mu2)}


data = [-2.1, -1.8, -2.3, -1.9, 3.1, 2.8, 3.3, 2.9, 3.0, -2.0]
posterior = infer(known_mixture, data, method="importance", num_samples=2000)
print(f"Cluster 1 mean: {posterior.mean('mu1'):.2f} (true ≈ -2)")
print(f"Cluster 2 mean: {posterior.mean('mu2'):.2f} (true ≈ 3)")

# ---------------------------------------------------------------------------
# 2. Unknown number of categories (with mem)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("2. Category learning with mem")
print("=" * 60)


@model
def category_learning(observations):
    """Each object belongs to a category; objects in the same category
    have the same mean value (persistent via mem).
    observations: list of measured values, one per object."""
    category_mean = mem(lambda cat: sample(Normal(0, 5)))
    n_cats = 3

    assignments = []
    for i, value in enumerate(observations):
        cat = sample(UniformDraw(list(range(n_cats))), name=f"cat_{i}")
        observe(Normal(category_mean(cat), 0.3), value)
        assignments.append(cat)

    return {
        "0_1_same": assignments[0] == assignments[1],
        "0_2_same": assignments[0] == assignments[2],
    }


posterior = infer(category_learning, [5.0, 5.1, -2.0], method="importance", num_samples=2000)
print(f"P(obj 0 and 1 same category) = {posterior.probability('0_1_same'):.3f}")
print(f"P(obj 0 and 2 same category) = {posterior.probability('0_2_same'):.3f}")

# ---------------------------------------------------------------------------
# 3. Infinite mixture model via DPmem
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("3. Dirichlet Process mixture (infinite components via DPmem)")
print("=" * 60)


@model
def dp_mixture(data):
    """Non-parametric clustering: the number of clusters is not fixed."""
    cluster_idx = [0]

    def new_cluster(_item):
        idx = cluster_idx[0]
        cluster_idx[0] += 1
        return idx

    get_cluster = DPmem(1.0, new_cluster)
    cluster_mean = mem(lambda c: sample(Normal(0, 10)))

    assignments = []
    for i, x in enumerate(data):
        c = get_cluster(i)
        observe(Normal(cluster_mean(c), 0.5), x)
        assignments.append(c)

    n_clusters = len(set(assignments))
    return n_clusters


posterior = infer(dp_mixture, [-2.0, -1.8, -2.1, 3.0, 3.2, 2.9], method="importance", num_samples=1000)
print(f"Number of clusters distribution: {posterior.histogram()}")

# ---------------------------------------------------------------------------
# 4. Bag of words topic model
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("4. Simple topic model")
print("=" * 60)


@model
def topic_model(document):
    """Two topics generate different word distributions.
    Given a document with certain words, infer its topic.
    Uses factor() to directly score each word under the topic's distribution."""
    topic = sample(UniformDraw(["science", "sports"]))

    science_words = {"neuron": 0.3, "brain": 0.3, "data": 0.2, "goal": 0.1, "ball": 0.1}
    sports_words = {"neuron": 0.05, "brain": 0.05, "data": 0.1, "goal": 0.4, "ball": 0.4}

    words = science_words if topic == "science" else sports_words
    word_list = list(words.keys())
    word_probs = list(words.values())
    dist = Categorical(word_list, word_probs)

    for word in document:
        factor(dist.log_prob(word))

    return topic


# Same model, different documents
posterior = infer(topic_model, ["brain", "neuron", "data"], method="importance", num_samples=5000)
print(f"P(science | 'brain neuron data') = {posterior.probability(lambda x: x == 'science'):.3f}")
print(f"P(sports | 'brain neuron data') = {posterior.probability(lambda x: x == 'sports'):.3f}")

posterior = infer(topic_model, ["goal", "ball", "goal"], method="importance", num_samples=5000)
print(f"P(science | 'goal ball goal') = {posterior.probability(lambda x: x == 'science'):.3f}")
print(f"P(sports | 'goal ball goal') = {posterior.probability(lambda x: x == 'sports'):.3f}")
