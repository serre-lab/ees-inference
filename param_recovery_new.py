import torch
import numpy as np

from sbi import utils as utils
from sbi.inference import SNPE, prepare_for_sbi
from sbi.utils.get_nn_models import posterior_nn

# import json
# from collections import OrderedDict
import matplotlib.pyplot as plt
import signal

class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException


# Vectorized implementation of the correlation coefficient computation
def vcorrcoef(x, y):
    centered_x = x - x.mean(1, keepdim=True)
    centered_y = y - y.mean()
    r_num = (centered_x * centered_y).sum(1)
    r_den = torch.sqrt((centered_x**2).sum(1) * (centered_y**2).sum())
    r = r_num / r_den
    return r

class LocalSimulator():
    def __init__(self, model, num_electrodes, electrode_index, device='cpu'):
        # amortized forward model
        self.model = model.to(device)

        # sets up the electrode activation pattern
        # this is currently a one-hot vector, but in principle can be any binary vector
        econfig = torch.tensor([0.]*num_electrodes)
        econfig[electrode_index] = 1.
        self.electrode_config = econfig.to(device)

        self.device = device


    ''' Version of the forward pass that we use with the local posterior version.
    theta --> incoming parameters as a pytorch tensor. 
    This is concatenated with the 'conditioned' electrode vector before it is 
    fed into the forward model
    '''
    def forward(self, theta):
        # self.model.eval()
        with torch.no_grad():
            inp_params = torch.cat([theta.to(self.device), self.electrode_config.expand(theta.shape[0],*self.electrode_config.size())],axis = 1)
            y_pred, _ = self.model(inp_params)
        return y_pred

    def eval(self, theta):
        self.model.eval()
        with torch.no_grad():
            inp_params = torch.cat([theta.to(self.device), self.electrode_config.expand(theta.shape[0],*self.electrode_config.size())],axis = 1)
            y_pred, _ = self.model(inp_params)
        return y_pred


class GlobalSimulator():
    def __init__(self, model, device='cpu'):
        # amortized forward model
        self.model = model.to(device)
        self.device = device

    def forward(self, theta):
        # self.model.eval()
        with torch.no_grad():
            y_pred, _ = self.model(theta.to(self.device))
        return y_pred

    def eval(self, theta):
        self.model.eval()
        with torch.no_grad():
            y_pred, _ = self.model(theta.to(self.device))
        return y_pred

class Inference():
    def __init__(self, 
        elec_encoding='onehot',
        num_rounds=2, 
        num_simulations=1024, 
        simulation_batch_size=1000,#1024, 
        training_batch_size=50,
        num_samples=10000, 
        filtering_ratio=0.1,
        num_proposals=5,
        timeout=600):

        self.elec_encoding = elec_encoding
        self.num_rounds = num_rounds
        self.num_simulations = num_simulations
        self.simulation_batch_size = simulation_batch_size
        self.training_batch_size = training_batch_size
        self.num_samples = num_samples
        self.filtering_ratio = filtering_ratio
        self.num_proposals = num_proposals
        self.timeout = timeout

    def train(self, simulator, target, _xy=None, height=None, width=None):
        # define a uniform prior
        # Ideally we would want to swap this out with the custom mized prior
        if self.elec_encoding == 'onehot':
            prior = utils.BoxUniform(
                low = torch.tensor([0., 0.]),
                high = torch.tensor([1., 1.])
            )
        else:
            from StimPrior import CustomStimPrior
            # prior = CustomStimPrior(
            #     _xy = _xy, 
            #     _lUniform = torch.tensor([0., 0.]), 
            #     _hUniform= torch.tensor([1., 1.]), _sigmaGauss=torch.Tensor([0.04, 0.1])
            # )
            prior = CustomStimPrior(
                _xy = _xy, 
                _lUniform = torch.tensor([0., 0.]), 
                _hUniform= torch.tensor([1., 1.]), _sigmaGauss=torch.Tensor([0.03, 0.08])
            )
            prior.visualize_prior_support(height, width)

            
            # prior = utils.BoxUniform(
            #     low = torch.tensor([0., 0., -1., -1.]),
            #     high = torch.tensor([1., 1., 1., 1.])
            # )

        # prepare the simulator and prior
        simulator, prior = prepare_for_sbi(simulator.forward, prior)

        # construct the network that will serve as our density estimator
        density_network = posterior_nn(model='maf')#, num_transforms=10)

        inference = SNPE(
            simulator, 
            prior, 
            density_estimator=density_network,
            show_progress_bars=True,
            simulation_batch_size=self.simulation_batch_size
        )

        # change the behavior of SIGALRM
        signal.signal(signal.SIGALRM, timeout_handler)
        # set timeout
        signal.alarm(self.timeout)
        try:
            proposal = None
            # posterior = inference(num_simulations=self.num_simulations, proposal=proposal, training_batch_size=self.training_batch_size)

            for k in range(self.num_rounds):
                posterior = inference(num_simulations=self.num_simulations, proposal=proposal, training_batch_size=self.training_batch_size)
                proposal = posterior.set_default_x(target)

        except TimeoutException:
            print('Inference terminated due to timeout')
            return None

        return posterior

    def sampling_proposals(self, simulator, posterior, target):
        theta = posterior.sample((self.num_samples,), x=target)
        log_probability = posterior.log_prob(theta, x=target)

        if self.elec_encoding == 'pos':
            max_log_prob = log_probability.max().item()
            min_log_prob = log_probability.min().item()

            fig = plt.figure()
            ax = fig.add_subplot()
            sc = ax.scatter(theta[:,2], theta[:,3], c=log_probability, cmap='seismic')
            ax.set_xlim((-1,1))
            # ax.set_ylim((-2.5,2.5))
            plt.colorbar(sc)
            fig.savefig('before_filtering.png')
            plt.close()

        # filtering by log_probability
        indices = log_probability.argsort(descending=True)
        n = int(self.num_samples * self.filtering_ratio)
        # n = self.num_samples
        theta = theta[indices][:n]
        log_probability = log_probability[indices][:n]

        if self.elec_encoding == 'pos':
            fig = plt.figure()
            ax = fig.add_subplot()
            sc = ax.scatter(theta[:,2], theta[:,3], c=log_probability, cmap='seismic')
            sc.set_clim(vmin=min_log_prob, vmax=max_log_prob)
            ax.set_xlim((-1,1))
            # ax.set_ylim((-2.5,2.5))
            plt.colorbar(sc)
            fig.savefig('after_filtering.png')
            plt.close()

        # x = simulator.forward(theta)
        x = simulator.eval(theta)
        return x, theta, log_probability

    def filtering_proposals(self, x, target, theta, log_probability, metric='corr'):
        if metric == 'corr':
            dist = 1 - vcorrcoef(x, target)
        elif metric == 'l1':
            dist = (x - target).abs().mean(dim=1)
        elif metric == 'l2':
            dist = ((x - target) ** 2).mean(dim=1)

        indices = dist.argsort(descending=False)
        x = x[indices][:self.num_proposals]
        theta = theta[indices][:self.num_proposals]
        log_probability = log_probability[indices][:self.num_proposals]
        dist = dist[indices][:self.num_proposals]

        return x, theta, log_probability, dist

    def pairplot(self, posterior, target, parameters):
        theta = posterior.sample((self.num_samples,), x=target)
        if self.elec_encoding == 'onehot':
            fig, ax = utils.pairplot(theta, limits = [[0.,1.], [0.,1.]], 
                        labels=['normalized frequency','normalized amplitude'], 
                        points=parameters[:2],
                        points_offdiag={'markersize':3, 'marker':'.'},
                        points_colors='r')
        else:
            fig, ax = utils.pairplot(theta, limits = [[0., 1.], [0, 1.], [-1.5,1.5], [-7,7]], 
                    labels=['normalized frequency','normalized amplitude', 'X', 'Y'], 
                    points=parameters,
                    points_offdiag={'markersize':3, 'marker':'.'},
                    points_colors='r')

        plt.show()
        
        return fig