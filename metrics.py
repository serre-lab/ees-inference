import torch
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

class Correlation(Metric):
    def __init__(self, dim, output_transform=lambda x: x):
        self._dim = dim
        super(Correlation, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self._sum_of_corrs = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output[0].detach(), output[1].detach()              
        
        centered_y_pred = y_pred - y_pred.mean(self._dim, keepdim=True)  
        centered_y = y - y.mean(self._dim, keepdim=True)  
        y_pred_std = torch.sqrt(torch.sum(centered_y_pred ** 2, self._dim))
        y_std = torch.sqrt(torch.sum(centered_y ** 2, self._dim))
        
        cov = torch.sum(centered_y_pred * centered_y, self._dim)
        corr = cov / (y_pred_std * y_std)     
        
        self._sum_of_corrs += corr.sum(0).mean().item()
        self._num_examples += y.shape[0]
    
    @sync_all_reduce("_sum_of_corrs", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("Correlation must have at least one example before it can be computed.")
        return self._sum_of_corrs / self._num_examples