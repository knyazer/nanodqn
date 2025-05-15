# ---------------------------------- set-up ----------------------------------
import torch, torch.nn as nn, torch.optim as optim, numpy as np, matplotlib, matplotlib.pyplot as plt

matplotlib.use("Agg")
torch.manual_seed(0)
np.random.seed(0)


def make_data(n=128):
    x = np.random.uniform(-5, 5, size=(n, 1)).astype(np.float32)
    y = np.sin(x) + 0.1 * np.random.randn(n, 1).astype(np.float32)
    return torch.as_tensor(x), torch.as_tensor(y)


x_train, y_train = make_data()


# -------------------------------- networks ----------------------------------
class MLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


class PriorHead(nn.Module):
    """f(x) = g_phi(x)  (frozen prior)  +  h_theta(x)  (trainable correction)"""

    def __init__(self, hidden=64):
        super().__init__()
        self.prior = MLP(hidden)
        for p in self.prior.parameters():
            p.requires_grad_(False)
        self.delta = MLP(hidden)

    def forward(self, x):
        return self.prior(x) + self.delta(x)


# ------------------------------- training -----------------------------------
def train(net, x, y, epochs=300, lr=1e-3):
    opt, loss = (
        optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr),
        nn.MSELoss(),
    )
    for _ in range(epochs):
        opt.zero_grad()
        l = loss(net(x), y)
        l.backward()
        opt.step()


def make_ensemble(cls, K=100):
    ens = []
    n = len(x_train)
    for _ in range(K):
        idx = torch.randint(0, n, (n,))  # bootstrap sample
        net = cls()
        train(net, x_train[idx], y_train[idx])
        ens.append(net)
    return ens


plain = make_ensemble(MLP)  # no prior
with_p = make_ensemble(PriorHead)  # randomized prior

# ------------------------------ evaluation ----------------------------------
grid = torch.linspace(-8, 8, 400).unsqueeze(1)


def stats(ens):
    preds = torch.stack([m(grid).squeeze(1) for m in ens])  # (K, N)
    return preds.mean(0).detach(), preds.std(0).detach()


μ_plain, σ_plain = stats(plain)
μ_prior, σ_prior = stats(with_p)

# -------------------------------- combined plot -----------------------------
fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(x_train, y_train, s=12, c="k", alpha=0.4, label="train data")

# ensemble means
ax.plot(grid, μ_plain, lw=2, c="C0", label="mean (no prior)")
ax.plot(grid, μ_prior, lw=2, c="C1", label="mean (with randomized prior)")

#  ±3 σ credible bands
ax.fill_between(
    grid.squeeze(),
    μ_plain - 3 * σ_plain,
    μ_plain + 3 * σ_plain,
    color="C0",
    alpha=0.18,
    label="±3 σ (no prior)",
)
ax.fill_between(
    grid.squeeze(),
    μ_prior - 3 * σ_prior,
    μ_prior + 3 * σ_prior,
    color="C1",
    alpha=0.18,
    label="±3 σ (with prior)",
)

# make the extra spread easier to see
ax.set_xlim(-8, 8)
ax.set_ylim(-3.4, 3.4)  # ↑ looser y-scale
ax.legend()
fig.tight_layout()

fig.savefig("combined_bootstrap_plot.png", dpi=180)  # rendered off-screen
