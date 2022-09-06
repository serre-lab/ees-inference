import torch
import numpy as np

from sbi import utils as utils
from sbi.inference import SNPE, prepare_for_sbi
from sbi.utils.get_nn_models import posterior_nn

# import json
# from collections import OrderedDict
import matplotlib.pyplot as plt
import signal
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import Circle
from matplotlib import cm
from scipy import stats

class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException


# Vectorized implementation of the correlation coefficient computation
def vcorrcoef(x, y):
    #import ipdb; ipdb.set_trace()
    centered_x = x - x.mean(1, keepdim=True)
    centered_y = y - y.mean()
    r_num = (centered_x * centered_y).sum(1)
    r_den = torch.sqrt((centered_x**2).sum(1) * (centered_y**2).sum())
    r = r_num / r_den
    return r

class Simulator():
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
        self.model.train()
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

class Inference():
    def __init__(self, 
        num_rounds=2, 
        num_simulations=1024, 
        simulation_batch_size=1000,#1024, 
        training_batch_size=50,
        num_samples=10000, 
        filtering_ratio=0.1,
        num_proposals=5,
        timeout=600):

        self.num_rounds = num_rounds
        self.num_simulations = num_simulations
        self.simulation_batch_size = simulation_batch_size
        self.training_batch_size = training_batch_size
        self.num_samples = num_samples
        self.filtering_ratio = filtering_ratio
        self.num_proposals = num_proposals
        self.timeout = timeout

    def train(self, simulator, target):
        # define a uniform prior
        # Ideally we would want to swap this out with the custom mized prior
        prior = utils.BoxUniform(
            low = torch.tensor([0., 0.]),
            high = torch.tensor([1., 1.])
        )

        # prepare the simulator and prior
        simulator, prior = prepare_for_sbi(simulator.forward, prior)
        
        # construct the network that will serve as our density estimator
        density_network = posterior_nn(model='maf')

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

        # max_log_prob = log_probability.max().item()
        # min_log_prob = log_probability.min().item()


        # filtering by log_probability
        indices = log_probability.argsort(descending=True)
        n = int(self.num_samples * self.filtering_ratio)
        theta = theta[indices][:n]
        log_probability = log_probability[indices][:n]



        # fig = plt.figure()
        # ax = fig.add_subplot()
        # sc = ax.scatter(theta[:,0], theta[:,1], c=log_probability, cmap='seismic')
        # ax.set_xlim((0,1))
        # ax.set_ylim((0,1))
        # plt.colorbar(sc)
        # fig.savefig('log_prob.png')
        # plt.close()


        # x = simulator.forward(theta)
        # corr = vcorrcoef(x, target)

        # fig = plt.figure()
        # ax = fig.add_subplot()
        # sc = ax.scatter(theta[:,0], theta[:,1], c=corr, cmap='seismic', vmin=-1, vmax=1)
        # ax.set_xlim((0,1))
        # ax.set_ylim((0,1))
        # plt.colorbar(sc)
        # fig.savefig('correlation.png')
        # plt.close()

        # select proposals based on correlation
        x = simulator.forward(theta)
        # x = simulator.eval(theta)
        corr = vcorrcoef(x, target)
        indices = corr.argsort(descending=True)
        x = x[indices][:self.num_proposals]
        theta = theta[indices][:self.num_proposals]
        log_probability = log_probability[indices][:self.num_proposals]
        corr = corr[indices][:self.num_proposals]

        resolution=100
        grid = torch.linspace(0,1,resolution)
        grid = torch.stack([grid[None,:].repeat(resolution,1), grid[:,None].repeat(1,resolution)], dim=2).view(-1,2)
        grid_x_noise = simulator.forward(grid)
        grid_x = simulator.eval(grid)
        grid_corr_noise = vcorrcoef(grid_x_noise, target)
        grid_corr = vcorrcoef(grid_x, target)

        X = np.linspace(0,1,resolution)
        Y = np.linspace(0,1,resolution)
        X, Y = np.meshgrid(X, Y)


        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        sc = ax.plot_surface(X, Y, grid_corr.view(resolution,resolution).numpy(), cmap=cm.coolwarm)
        point = Circle((theta[0,0].item(), theta[0,1].item()), 0.02, alpha=1, color='darkviolet', lw=2, fc="None")
        ax.add_patch(point)
        art3d.pathpatch_2d_to_3d(point, z=-1, zdir="z")
        cset = ax.contour(X, Y, grid_corr.view(resolution,resolution).numpy(), zdir='z', offset=-1, cmap=cm.coolwarm)

        ax.view_init(35, 35)
        ax.set_xlim((0,1))
        ax.set_ylim((0,1))
        ax.set_zlim(bottom=-1)
        ax.set_xlabel('Freq')
        ax.set_ylabel('Amp')
        ax.set_zlabel('Corr')
        plt.colorbar(sc)
        fig.savefig('3d_correlation.png')
        plt.close()


        fig = plt.figure()
        ax = fig.add_subplot()
        sc = ax.contourf(X, Y, (grid_corr_noise - grid_corr).abs().view(resolution,resolution).numpy(), cmap=cm.coolwarm, vmax=0.5, vmin=0)
        ax.set_xlim((0,1))
        ax.set_ylim((0,1))
        ax.set_xlabel('Freq')
        ax.set_ylabel('Amp')

        plt.colorbar(sc)
        fig.savefig('3d_correlation_diff.png')
        plt.close()

        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # sc = ax.plot_surface(X, Y, (grid_corr_noise - grid_corr).abs().view(resolution,resolution).numpy(), cmap=cm.coolwarm)
        # cset = ax.contourf(X, Y, (grid_corr_noise - grid_corr).abs().view(resolution,resolution).numpy(), zdir='x', offset=-0.1, cmap=cm.coolwarm)
        # cset = ax.contourf(X, Y, (grid_corr_noise - grid_corr).abs().view(resolution,resolution).numpy(), zdir='y', offset=-0.1, cmap=cm.coolwarm)
        # ax.view_init(35, 35)
        # ax.set_xlim((-0.1,1))
        # ax.set_ylim((-0.1,1))
        # ax.set_zlim((0,0.5))
        # ax.set_xlabel('Freq')
        # ax.set_ylabel('Amp')
        # ax.set_zlabel('Abs Diff Corr')
        # plt.colorbar(sc)
        # fig.savefig('3d_correlation_diff.png')
        # plt.close()


        l1_noise = (grid_x_noise - target).abs().mean(-1)
        l1 = (grid_x - target).abs().mean(-1)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        sc = ax.plot_surface(X, Y, l1.view(resolution,resolution).numpy(), cmap=cm.coolwarm)
        point = Circle((theta[0,0].item(), theta[0,1].item()), 0.02, alpha=1, color='darkviolet', lw=2, fc="None")#, zorder=20)
        ax.add_patch(point)
        art3d.pathpatch_2d_to_3d(point, z=0, zdir="z")
        cset = ax.contour(X, Y, l1.view(resolution,resolution).numpy(), zdir='z', offset=0, cmap=cm.coolwarm)#, alpha=0.9, zorder=-1)

        ax.view_init(35, 35)
        ax.set_xlim((0,1))
        ax.set_ylim((0,1))
        ax.set_zlim(bottom=0)
        ax.set_xlabel('Freq')
        ax.set_ylabel('Amp')
        ax.set_zlabel('L1')

        plt.colorbar(sc)
        fig.savefig('3d_l1.png')
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot()
        sc = ax.contourf(X, Y, (l1_noise - l1).abs().view(resolution,resolution).numpy(), cmap=cm.coolwarm, vmax=0.5, vmin=0)
        ax.set_xlim((0,1))
        ax.set_ylim((0,1))
        ax.set_xlabel('Freq')
        ax.set_ylabel('Amp')

        plt.colorbar(sc)
        fig.savefig('3d_l1_diff.png')
        plt.close()

        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # sc = ax.plot_surface(X, Y, (l1_noise - l1).abs().view(resolution,resolution).numpy(), cmap=cm.coolwarm)
        # cset = ax.contourf(X, Y, (l1_noise - l1).abs().view(resolution,resolution).numpy(), zdir='x', offset=-0.1, cmap=cm.coolwarm)
        # cset = ax.contourf(X, Y, (l1_noise - l1).abs().view(resolution,resolution).numpy(), zdir='y', offset=-0.1, cmap=cm.coolwarm)
        # ax.view_init(35, 35)
        # ax.set_xlim((-0.1,1))
        # ax.set_ylim((-0.1,1))
        # ax.set_zlim((0,0.2))
        # ax.set_xlabel('Freq')
        # ax.set_ylabel('Amp')
        # ax.set_zlabel('Abs Diff L1')
        # plt.colorbar(sc)
        # fig.savefig('3d_l1_diff.png')
        # plt.close()





        # theta = posterior.sample((self.num_samples,), x=target)
        # log_probability = posterior.log_prob(theta, x=target)


        # # filtering by log_probability
        # indices = log_probability.argsort(descending=True)
        # n = int(self.num_samples * self.filtering_ratio)
        # theta = theta[indices][:n]
        # log_probability = log_probability[indices][:n]

        # # select proposals based on correlation
        # # x = simulator.forward(theta)
        # x = simulator.eval(theta)
        # corr = vcorrcoef(x, target)
        # indices = corr.argsort(descending=True)
        # x = x[indices][:self.num_proposals]
        # theta = theta[indices][:self.num_proposals]
        # log_probability = log_probability[indices][:self.num_proposals]
        # corr = corr[indices][:self.num_proposals]


        # grid = torch.linspace(0,1,100)
        # grid = torch.stack([grid[None,:].repeat(100,1), grid[:,None].repeat(1,100)], dim=2).view(-1,2)
        # # grid_x = simulator.forward(grid)
        # grid_x = simulator.eval(grid)
        # grid_corr = vcorrcoef(grid_x, target)

        # X = np.linspace(0,1,100)
        # Y = np.linspace(0,1,100)
        # X, Y = np.meshgrid(X, Y)


        # # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # # ax.plot(theta[0,0], theta[0,1], corr[0], 'go', alpha=1, markersize=20)
        # sc = ax.plot_surface(X, Y, grid_corr.view(100,100).numpy(), cmap=cm.coolwarm)

        # point = Circle((theta[0,0].item(), theta[0,1].item()), 0.02, alpha=1, color='darkviolet', lw=2, fc="None")
        # ax.add_patch(point)
        # art3d.pathpatch_2d_to_3d(point, z=-1, zdir="z")

        # cset = ax.contour(X, Y, grid_corr.view(100,100).numpy(), zdir='z', offset=-1, cmap=cm.coolwarm)

        # ax.view_init(35, 35)
        # ax.set_xlim((0,1))
        # ax.set_ylim((0,1))
        # ax.set_zlim(bottom=-1)
        # ax.set_xlabel('Freq')
        # ax.set_ylabel('Amp')
        # ax.set_zlabel('Corr')
        # plt.colorbar(sc)
        # fig.savefig('3d_correlation_eval.png')
        # plt.close()


        # l1 = (grid_x - target).abs().mean(-1)
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # sc = ax.plot_surface(X, Y, l1.view(100,100).numpy(), cmap=cm.coolwarm)
        # point = Circle((theta[0,0].item(), theta[0,1].item()), 0.02, alpha=1, color='darkviolet', lw=2, fc="None")#, zorder=20)
        # ax.add_patch(point)
        # art3d.pathpatch_2d_to_3d(point, z=0, zdir="z")
        # cset = ax.contour(X, Y, l1.view(100,100).numpy(), zdir='z', offset=0, cmap=cm.coolwarm)#, alpha=0.9, zorder=-1)

        # ax.view_init(35, 35)
        # ax.set_xlim((0,1))
        # ax.set_ylim((0,1))
        # ax.set_zlim(bottom=0)
        # ax.set_xlabel('Freq')
        # ax.set_ylabel('Amp')
        # ax.set_zlabel('L1')

        # plt.colorbar(sc)
        # fig.savefig('3d_l1_eval.png')
        # plt.close()


        return x, theta, log_probability, corr

    def pairplot(self, posterior, target, parameters):
        theta = posterior.sample((self.num_samples,), x=target)
        fig, ax = utils.pairplot(theta, limits = [[0.,1.], [0.,1.]], 
                    labels=['normalized frequency','normalized amplitude'], 
                    points=parameters[:2],
                    points_offdiag={'markersize':3, 'marker':'.'},
                    points_colors='r')
        #plt.show()

        return fig

    def fancypairplot(self, posterior, target, parameters, select_mean=True, stars=None, contour=False):
        theta = posterior.sample((10000,), x=target)

        N = 100
        bins = np.linspace(0, 1, N)
        
        # Set up the axes with gridspec
        fig = plt.figure(figsize=(6, 6))
        grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
        
        main_ax = fig.add_subplot(grid[1:, :-1])
        y_hist = fig.add_subplot(grid[1:, -1])
        x_hist = fig.add_subplot(grid[0, :-1])

        X, Y = np.mgrid[0:1:100j, 0:1:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([theta[:,1].numpy(), theta[:, 0].numpy()])
        kernel = stats.gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)

        grad = np.gradient(Z)

        def compute_path(grad, pos, n_steps=50, step_size=50):
            if not select_mean:
                p_x, p_y = [pos[0]], [pos[1]]
            else:
                from numpy import unravel_index
                idx = unravel_index(Z.argmax(), Z.shape)
                #p_x, p_y = [int(theta[:,1].mode().values.item() * 100)], [int(theta[:,0].mode().values.item() * 100)]
                p_x, p_y = [idx[0]], [idx[1]]

            for k in range(n_steps):
                g_x, g_y = grad[0][p_x[-1], p_y[-1]], grad[1][p_x[-1], p_y[-1]]
                p_x.append(max(min(int(p_x[-1] + g_x * step_size), 99),0))
                p_y.append(max(min(int(p_y[-1] + g_y * step_size),99),0))
                if np.sqrt((p_x[-1] - p_x[-2])**2 + (p_y[-1] - p_y[-2])**2) < 2:
                    #reached fixed point
                    break
                if (k+1)%10 == 0:
                    step_size /= 2.
            return np.array(p_x)/100., np.array(p_y)/100.

       
        # Covariance
        main_ax.imshow(np.rot90(Z), cmap=plt.cm.Oranges, extent=[0., 1., 0., 1.], zorder=1)
        if contour:
            cs = main_ax.contour(X, Y, Z, 3, colors='white')
            pth = cs.collections[2].get_paths()[0]
            xs = pth.vertices[::15, 0]
            ys = pth.vertices[::15, 1]
            main_ax.scatter(xs, ys, c='w', marker='*', edgecolor='k', s=48, zorder=3)
            '''
            for i in range(xs.shape[0]):
                if i < xs.shape[0]/2:
                    main_ax.annotate('P{}'.format(i), (xs[i]-0.01, ys[i]+0.03), fontsize=14)
                else:
                    main_ax.annotate('P{}'.format(i), (xs[i]-0.01, ys[i]-0.03), fontsize=14)
            '''
            path_x, path_y = xs, ys

        # Mark just the supplied points
        if (type(stars) != type(None)) and (contour == False):
            main_ax.scatter(stars[:, 1], stars[:, 0], c='w', marker='*', s=48, edgecolor='k',zorder=3)
        elif contour == False:
            # find paths/contours
            path_x, path_y = compute_path(grad, [int(parameters[1].item() * 100), int(parameters[0].item() * 100)])
            path_x = path_x[::10]
            path_y = path_y[::10]
     
            main_ax.plot(path_x, path_y, '--w', zorder=2)
            main_ax.scatter(path_x, path_y, c='w', marker='*', s=48, edgecolor='k',zorder=3)

            for i in range(path_x.shape[0]):
                main_ax.annotate('P{}'.format(i), (path_x[i]+0.01, path_y[i]+0.01), fontsize=14)

            #main_ax.plot( np.array([pos_y, pos_y + g_y*50])/100., np.array([pos_x, pos_x + g_x*50])/100., c='r')

        main_ax.spines['right'].set_visible(False)
        main_ax.spines['top'].set_visible(False)
        main_ax.yaxis.set_ticks_position('left')
        main_ax.xaxis.set_ticks_position('bottom')
        main_ax.set_xticks([0., 1.])
        main_ax.set_xticklabels([0., 1.])

        main_ax.set_yticks([0., 1.])
        main_ax.set_yticklabels([0., 1.])

        main_ax.set_xlabel('normalized amplitude')
        main_ax.set_ylabel('normalized frequency')
 

        # Marginals on the attached axes
        x_hist.hist( theta[:,1].numpy(), density=True, bins=bins, alpha=0.75, color='tab:brown', orientation='vertical')
        kde = stats.gaussian_kde(theta[:,1].numpy())
        #x_hist.plot(bins, kde(bins), c='crimson', linestyle='--', linewidth=2) 
        #x_hist.invert_yaxis()
        x_hist.spines['right'].set_visible(False)
        x_hist.spines['top'].set_visible(False)
        x_hist.spines['left'].set_visible(False)
        x_hist.set_xticks([])
        x_hist.set_yticks([])

        y_hist.hist(theta[:,0].numpy(), density=True, bins=bins, color='tab:brown', alpha=0.75, orientation='horizontal')
        kde = stats.gaussian_kde(theta[:,0].numpy())
        #y_hist.plot(kde(bins), bins, c='crimson', linestyle='--', alpha=0.5, linewidth=2) 
 
        #y_hist.invert_xaxis()
        y_hist.spines['top'].set_visible(False)
        y_hist.spines['bottom'].set_visible(False)
        y_hist.spines['right'].set_visible(False)
        y_hist.set_xticks([])
        y_hist.set_yticks([])

        plt.savefig('posterior_sensitivity.png', bbox_inches='tight')
        #plt.show()

        if contour:
            return path_x, path_y
        elif type(stars) != type(None): 
            return None, None
        else:
            return path_x, path_y
