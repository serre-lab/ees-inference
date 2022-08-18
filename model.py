import torch
from torch import nn
import torch.nn.functional as F

from hydra.utils import instantiate

class Stim2EMG(nn.Module):
    def __init__(self, network, optimizer, loss):
        super(Stim2EMG, self).__init__()
        self.emb = instantiate(network.embedding)
        self.core = instantiate(network.core)
        self.readout = instantiate(network.readout)
        
        self.optimizer = instantiate(optimizer, self.parameters())
        self.criterion = instantiate(loss)

    def forward(self, x):
        h = self.emb(x)
        h = self.core(h)
        y_pred = self.readout(h)

        return y_pred, h

    def _update(self, engine, batch):
        self.train()
        self.optimizer.zero_grad()
        
        x, y = batch
        y_pred, _ = self(x)
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _inference(self, engine, batch):
        self.eval()

        with torch.no_grad():
            x, y = batch
            y_pred, z = self(x)

            return {'x': x, 'y_pred': y_pred, 'y': y, 'z': z}

    def _stim_inference(self, engine, batch):
        raise NotImplementedError


class Stim2StimEMG(nn.Module):
    def __init__(self, network, optimizer, loss):
        super(Stim2StimEMG, self).__init__()
        
        self.mu_prior = nn.Parameter(torch.zeros(1, network.latent_size))
        self.logvar_prior = nn.Parameter(torch.zeros(1, network.latent_size))
        
        self.encoder = instantiate(network.encoder)
        self.mu = nn.Linear(network.encoder.hidden_sizes[-1], network.latent_size)
        self.logvar = nn.Linear(network.encoder.hidden_sizes[-1], network.latent_size)
        self.decoder = instantiate(network.decoder)
        self.readout = instantiate(network.readout)

        self.optimizer = instantiate(optimizer, self.parameters())

        self.beta = loss.beta
        self.stim_criterion = instantiate(loss['stim'])
        self.emg_criterion = instantiate(loss['emg'])

    def reparameterize(self, mu, logvar, num_samples=1):
        sigma = logvar.mul(0.5).exp_()
        if num_samples > 1:
            mu = mu.repeat(num_samples, 1)
            sigma = sigma.repeat(num_samples, 1)
            
        eps = torch.randn_like(sigma)
        return eps.mul(sigma).add_(mu)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)

        # sampling
        z = self.reparameterize(mu, logvar)

        x_pred = self.decoder(z)
        y_pred = self.readout(z)

        return y_pred, x_pred, mu, logvar, z

    def kl_criterion(self, mu, logvar):
        sigma_prior = self.logvar_prior.mul(0.5).exp() 
        sigma = logvar.mul(0.5).exp() 
        
        kld = torch.log(sigma_prior/sigma) + (torch.exp(logvar) + (mu - self.mu_prior)**2)/(2*torch.exp(self.logvar_prior)) - 1/2
        return kld.mean(0).sum()

    def _update(self, engine, batch):
        self.train()
        self.optimizer.zero_grad()
        
        x, y = batch
        y_pred, x_pred, mu, logvar, _ = self(x)

        kl_loss = self.kl_criterion(mu, logvar)
        # stim_loss = self.stim_criterion(x_pred, x)
        
        stim_loss = self.stim_criterion(F.sigmoid(x_pred[:,:2]), x[:,:2]) \
                    + F.nll_loss(F.log_softmax(x_pred[:,2:], dim=1), torch.argmax(x[:,2:],dim=1))
        # stim_loss = self.stim_criterion(F.sigmoid(x_pred[:,:2]), x[:,:2]) \
        #             + F.nll_loss(torch.log(F.gumbel_softmax(x_pred[:,2:], dim=1)), torch.argmax(x[:,2:],dim=1))

        emg_loss = self.emg_criterion(y_pred, y)
        loss = emg_loss + stim_loss + self.beta * kl_loss
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _inference(self, engine, batch):
        self.eval()

        with torch.no_grad():
            x, y = batch
            y_pred, _, _, _, z = self(x)

            return {'x': x, 'y_pred': y_pred, 'y': y, 'z': z}