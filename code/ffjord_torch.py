from torchdyn.models import CNF, NeuralDE
import torch
from torch.distributions import (
    MultivariateNormal,
    Uniform,
    TransformedDistribution,
    SigmoidTransform,
    Categorical,
)


def hutch_trace(x_out, x_in, noise=None, **kwargs):
    """Hutchinson's trace Jacobian estimator, O(1) call to autograd"""
    jvp = torch.autograd.grad(x_out, x_in, noise, create_graph=True)[0]
    trJ = torch.einsum("bi,bi->b", jvp, noise)
    return trJ


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

DATA_DIM = 8
prior = MultivariateNormal(
    torch.zeros(DATA_DIM).to(device), torch.eye(DATA_DIM).to(device)
)
noise_dist = MultivariateNormal(
    torch.zeros(DATA_DIM).to(device), torch.eye(DATA_DIM).to(device)
)

# f is the neural network.
f = torch.nn.Module()
cnf = CNF(f, trace_estimator=hutch_trace, noise_dist=noise_dist)
nde = NeuralDE(
    cnf,
    solver="dopri5",
    s_span=torch.linspace(0, 1, 2),
    sensitivity="adjoint",
    atol=1e-4,
    rtol=1e-4,
)
