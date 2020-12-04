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
    def __init__(
        self,
        model: nn.Module,
        device=torch.device("cpu"),
        ddim=8,
        lr=2e-3,
        weight_decay=1e-5,
    ):
        super().__init__()
        self.model = model
        self.iters = 0
        self.lr = lr
        self.weight_decay = weight_decay

        # base distribution
        self.prior = MultivariateNormal(
            torch.zeros(ddim).to(device), torch.eye(ddim).to(device)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.iters += 1
        x = batch[0]
        xtrJ = self.model(x)

        # logp(z_S) = logp(z_0) - \int_0^S trJ
        logprob = self.prior.log_prob(xtrJ[:, 1:]).to(x) - xtrJ[:, 0]
        loss = -torch.mean(logprob)
        self.model.nfe = 0
        return loss

    def configure_optimizers(self):
        return AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


class DataModule(pl.LightningDataModule):
    def __init__(self, latents, batch_size):
        super().__init__()
        self.data = latents
        self.batch_size = batch_size
        assert latents.shape[1] == 8


def run_ffjord(
    datamodule,
    device,
    weight_decay=1e-5,
    lr=2e-3,
    ddim=8,
    max_epochs=100,
    train_batches=10,
):

    # needed for hutchinson trace estimator
    noise_dist = MultivariateNormal(
        torch.zeros(ddim).to(device), torch.eye(ddim).to(device)
    )

    # f is the neural network of your choice.
    # SoftPlus is a smooth approximation to ReLU (ODEs require smoothness).
    f = nn.Sequential(
        nn.Linear(ddim, 64),
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
    model = nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), nde).to(device)
    learn = Learner(model, device=device, weight_decay=weight_decay, lr=lr, ddim=ddim)
    trainer = pl.Trainer(max_epochs=max_epochs, limit_train_batches=train_batches)
    trainer.fit(learn, datamodule=datamodule)
    return learn.model
