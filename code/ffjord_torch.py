from torchdyn.models import CNF, NeuralDE, Augmenter
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.distributions import MultivariateNormal


def hutch_trace(x_out, x_in, noise=None, **kwargs):
    """Hutchinson's trace Jacobian estimator, O(1) call to autograd"""
    jvp = torch.autograd.grad(x_out, x_in, noise, create_graph=True)[0]
    trJ = torch.einsum("bi,bi->b", jvp, noise)
    return trJ


class Learner(pl.LightningModule):
    def __init__(self, _model: nn.Module):
        super().__init__()
        self.model = _model
        self.iters = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.iters += 1
        x = batch[0]
        xtrJ = self.model(x)

        # logp(z_S) = logp(z_0) - \int_0^S trJ
        logprob = prior.log_prob(xtrJ[:, 1:]).to(x) - xtrJ[:, 0]
        loss = -torch.mean(logprob)
        self.model.nfe = 0
        return loss

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


# INPUTS
DATA_DIM = 8
LR = 2e-3  # learning rate for optimizer
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

prior = MultivariateNormal(
    torch.zeros(DATA_DIM).to(DEVICE), torch.eye(DATA_DIM).to(DEVICE)
)
noise_dist = MultivariateNormal(
    torch.zeros(DATA_DIM).to(DEVICE), torch.eye(DATA_DIM).to(DEVICE)
)

# f is the neural network of your choice.
# SoftPlus is a smooth approximation to ReLU (ODEs require smoothness).
f = nn.Sequential(
    nn.Linear(DATA_DIM, 64),
    nn.Softplus(),
    nn.Linear(64, 64),
    nn.Softplus(),
    nn.Linear(64, 64),
    nn.Softplus(),
    nn.Linear(64, 2),
)
cnf = CNF(f, trace_estimator=hutch_trace, noise_dist=noise_dist)
nde = NeuralDE(
    cnf,
    solver="dopri5",
    s_span=torch.linspace(0, 1, 2),
    sensitivity="adjoint",
    atol=1e-4,
    rtol=1e-4,
)
model = nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), nde).to(DEVICE)
