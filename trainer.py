import torch
import torch.nn.functional as F
import numpy as np
import os
import logging

from ignite.engine import Engine, DeterministicEngine, Events
from ignite.handlers import ModelCheckpoint, Checkpoint, global_step_from_engine
from ignite.utils import setup_logger
from ignite.metrics import Loss, RunningAverage
from hydra.utils import instantiate
from utils.handler import output_transform, prepare_batch, switch_batch, fetch_last_checkpoint_model_filename, save_activation
from utils.visualization import tsne

from param_recovery import Simulator, Inference
import json
from collections import OrderedDict
import pickle
from scipy.optimize import differential_evolution
import tqdm
from joblib import Parallel, delayed

class MlObj():
    def __init__(self, 
            model,
            all_params,
            all_targets,
            index,
            dim = None):

        self.model = model
        #self.model.eval()
        self.target = all_targets[index]
        self.econfig = all_params[index,2:]
        self.param_config = all_params[index, :2]
        self._prm = dim

        self._dim = 0
        self.mse = torch.nn.MSELoss()

    def objfn(self, params):
        if self._prm == None:
            x = torch.cat([torch.tensor(params), self.econfig]).float()
        elif self._prm == 0:
            x = torch.cat([torch.tensor(params), self.param_config[1].unsqueeze(-1), self.econfig]).float()
        else:
            x = torch.cat([self.param_config[0].unsqueeze(-1), torch.tensor(params), self.econfig]).float()
 
        y_pred, _ = self.model(x)

        '''
        ## USE CORR AS A METRIC
        y_pred = y_pred.detach()
        y = self.target.detach()        

        #import ipdb; ipdb.set_trace()
        centered_y_pred = y_pred - y_pred.mean(self._dim, keepdim=True)  
        centered_y = y - y.mean(self._dim, keepdim=True)  
        y_pred_std = torch.sqrt(torch.sum(centered_y_pred ** 2, self._dim))
        y_std = torch.sqrt(torch.sum(centered_y ** 2, self._dim))
        
        cov = torch.sum(centered_y_pred * centered_y, self._dim)
        corr = cov / (y_pred_std * y_std)     
        
        _sum_of_corrs = corr.sum(0).mean().item()
        _num_examples = y.shape[0]
        '''

        #return self.mse(y_pred, self.target).item()
        return self.model.criterion(y_pred, self.target).item()
        #return -1 * (_sum_of_corrs / _num_examples)

class Trainer():
    def __init__(self,
            num_epochs,
            metrics,
            device,
            deterministic,
            eval_interval,
            checkpoint_dir,
            num_rounds, 
            num_simulations, 
            simulation_batch_size,
            training_batch_size,
            num_samples,
            filtering_ratio,
            num_proposals,
            timeout):
        
        # assert device in ['cpu', 'cuda']

        self.num_epochs = num_epochs
        self.eval_interval = eval_interval
        self.device = device
        self.deterministic = deterministic
        self.checkpoint_dir = checkpoint_dir
        
        self.num_rounds = num_rounds 
        self.num_simulations = num_simulations
        self.simulation_batch_size = simulation_batch_size
        self.training_batch_size = training_batch_size
        self.num_samples = num_samples
        self.filtering_ratio = filtering_ratio
        self.num_proposals = num_proposals
        self.timeout = timeout

        self.metrics = {k:instantiate(v, output_transform=output_transform) for (k, v) in metrics.items()}

        
    def train(self, model, datamodule):
        model = model.to(self.device)
        # datamodule.prepare_data()
        # datamodule.setup(self.fold_idx) # 10 fold cv

        train_dataloader = datamodule.train_dataloader()
        valid_dataloader = datamodule.valid_dataloader()

        trainer = Engine(model._update) if not self.deterministic else DeterministicEngine(model._update)
        evaluator = Engine(model._inference)
        # trainer.logger.disabled = True
        # trainer.logger = setup_logger("Trainer")
        evaluator.logger = setup_logger("Evaluator")

        # attach running average to trainer
        RunningAverage(output_transform=lambda x: x).attach(trainer, "running_avg_loss")
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_running_avg_metrics(engine):
            print("Epoch[{}] Running avg loss: {:.2f}".format(engine.state.epoch, engine.state.metrics['running_avg_loss']))

        # attach metrics to evaluator
        for name, metric in self.metrics.items():
            metric.attach(evaluator, name)

        # attach validation step to trainer
        @trainer.on(Events.EPOCH_COMPLETED(every=self.eval_interval))
        def log_validation_results(engine):
            evaluator.run(valid_dataloader)
            metrics = evaluator.state.metrics
            log = 'Stage: %s | Epoch: %d' % ('valid', engine.state.epoch)
            for name in self.metrics.keys():
                log = '%s | %s: %f' % (log, name, metrics[name])
            print(log)

        # attach checkpointer to evaluator
        checkpointer = ModelCheckpoint(
            self.checkpoint_dir,
            n_saved=None,
            filename_prefix="",
            score_function=lambda engine: engine.state.metrics["corr"],
            score_name="valid_corr",
            global_step_transform=global_step_from_engine(trainer),
        )
        to_save = {
            'model': model, 
            'trainer': trainer
        }
        evaluator.add_event_handler(Events.COMPLETED, checkpointer, to_save)

        # attach switch_batch for tensor device conversion
        trainer.add_event_handler(Events.ITERATION_STARTED, switch_batch, self.device)
        evaluator.add_event_handler(Events.ITERATION_STARTED, switch_batch, self.device)

        # run trainer
        trainer.run(train_dataloader, max_epochs=self.num_epochs)

    def eval(self, model, datamodule, stage, checkpointer=None, model_save_path=None, visualization=None):
        assert stage in ['train', 'valid', 'test']
        model = model.to(self.device)
        # datamodule.prepare_data()
        # datamodule.setup(self.fold_idx) # 10 fold cv

        loader = {
            'train': datamodule.train_dataloader, 
            'valid': datamodule.valid_dataloader,
            'test': datamodule.test_dataloader
        }[stage]()
        
        evaluator = Engine(model._inference)
        
        # load a trained model if checkpointer or model_save_path is given
        if model_save_path is not None:
            checkpointer = fetch_last_checkpoint_model_filename(model_save_path)
        
        if checkpointer is not None:
            print(checkpointer)
            to_load = {"model": model}
            checkpoint = torch.load(checkpointer)
            Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

        evaluator.logger = setup_logger("Evaluator")

        # attach metrics to the evaluator
        for name, metric in self.metrics.items():
            metric.attach(evaluator, name)

        @evaluator.on(Events.EPOCH_COMPLETED)
        def log_results(engine):
            metrics = engine.state.metrics
            log = 'Stage: %s' % stage
            for name in self.metrics.keys():
                log = '%s | %s: %f' % (log, name, metrics[name])
            print(log)
        
        # attach switch_batch for tensor device conversion
        evaluator.add_event_handler(Events.ITERATION_STARTED, switch_batch, self.device)
        if visualization is not None:
            activation_dict = {'x': [], 'y_pred': [], 'y': [], 'z': []}
            evaluator.add_event_handler(Events.ITERATION_COMPLETED, save_activation, activation_dict)

        # run evaluator
        evaluator.run(loader)

        if visualization == 'tsne':
            z = torch.cat(activation_dict['z'], dim=0).cpu().data.numpy()
            x = torch.cat(activation_dict['x'], dim=0).cpu().data.numpy()
            tsne(z, x, datamodule.ees_channels, os.getcwd())


    def get_metrics(self, model, datamodule, stage, electrode_index, target_index, checkpointer=None, model_save_path=None):

        model = model.to(self.device)
        all_params = datamodule.test_dataset.tensors[0]
        all_targets = datamodule.test_dataset.tensors[1]
        
        # load a trained model if checkpointer or model_save_path is given
        if model_save_path is not None:
            checkpointer = fetch_last_checkpoint_model_filename(model_save_path)
        
        if checkpointer is not None:
            print(checkpointer)
            to_load = {"model": model}
            checkpoint = torch.load(checkpointer, map_location=self.device)
            Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

        model = model.eval()
        #import ipdb; ipdb.set_trace()
        preds, _ = model(all_params)
        rand_preds = torch.rand(preds.shape)

        err = torch.abs(preds - all_targets)
        rand_err = torch.abs(rand_preds - all_targets)

        _err = err.detach().cpu().numpy()
        _randerr = rand_err.detach().cpu().numpy()

        X = {'channels': [x for x in datamodule.selected_muscles], 'err': _err, 'randerr': _randerr}
        pickle.dump(X, open('l1_errors.p', 'wb'))


    def MLEV0(self, model, datamodule, stage, checkpointer=None, model_save_path=None):

        model = model.to(self.device)
        all_params = datamodule.valid_dataset.tensors[0]
        all_targets = datamodule.valid_dataset.tensors[1]

        filt_idx = torch.where(torch.sum(all_targets, dim=1) > 1.)[0]
        all_params = all_params[filt_idx]
        all_targets = all_targets[filt_idx]

        # load a trained model if checkpointer or model_save_path is given
        if model_save_path is not None:
            checkpointer = fetch_last_checkpoint_model_filename(model_save_path)

            ## THELMA
            #checkpointer = '/media/data_cifs/projects/prj_deepspine/minju/deep-spine-ignite-parquet/outputs/mlp/sheep_20210604/both/2021-06-04_13-42-57/models/checkpoint_300_valid_corr=0.9351.pt' #checkpoint_100_valid_corr=0.9142.pt'

            ## LOUISE
            #checkpointer = '/media/data_cifs/projects/prj_deepspine/minju/deep-spine-ignite-parquet/outputs/mlp/sheep_20210610/both/2021-06-10_12-03-42/models/checkpoint_300_valid_corr=0.9421.pt'

        if checkpointer is not None:
            print(checkpointer)
            to_load = {"model": model}
            checkpoint = torch.load(checkpointer, map_location=self.device)
            Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

        #model = model.eval()
        bounds = [(0., 0.9), (0., 0.9)]

        freq_true, freq_rec, amp_true, amp_rec = [], [], [], []

        def recover(all_params, all_targets, index):
            #bounds = [(0.075, 0.75), (0.2, 0.75)]
            my_class = MlObj(model, all_params, all_targets, index)
            output = differential_evolution(my_class.objfn, bounds)
            return output.x

        results = Parallel(n_jobs=8)(delayed(recover)(all_params, all_targets, i) for i in tqdm.tqdm(range(all_targets.shape[0]))) 

        for idx, res in enumerate(results):
            freq_rec.append(res[0])
            amp_rec.append(res[1])
            freq_true.append(all_params[idx,0].item())
            amp_true.append(all_params[idx,1].item())

        '''
        for index in tqdm.tqdm(range(all_targets.shape[0])):
            my_class = MlObj(model, all_params, all_targets, index)
            output = differential_evolution(my_class.objfn, bounds)
            freq_rec.append(output.x[0])
            amp_rec.append(output.x[1])
            freq_true.append(all_params[index,0].item())
            amp_true.append(all_params[index,1].item())
        '''

        my_res = {'freq_true': freq_true, 'freq_rec': freq_rec, 'amp_true': amp_true, 'amp_rec': amp_rec}
        pickle.dump(my_res, open('MLEOutput.p','wb'))

    def MLE(self, model, datamodule, stage, checkpointer=None, model_save_path=None, repeats=5):

        model = model.to(self.device)
        all_params = datamodule.valid_dataset.tensors[0]
        all_targets = datamodule.valid_dataset.tensors[1]

        import ipdb; ipdb.set_trace()

        filt_idx = torch.where(torch.sum(all_targets, dim=1) > 1.)[0]
        all_params = all_params[filt_idx]
        all_targets = all_targets[filt_idx]

        # load a trained model if checkpointer or model_save_path is given
        if model_save_path is not None:
            checkpointer = fetch_last_checkpoint_model_filename(model_save_path)

        if checkpointer is not None:
            print(checkpointer)
            to_load = {"model": model}
            checkpoint = torch.load(checkpointer, map_location=self.device)
            Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

        #model = model.eval()
        freq_true, freq_rec, amp_true, amp_rec = [], [], [], []

        def recover(all_params, all_targets, index, dim):
            if dim == 0:
                bounds = [(0.075, 0.9)] #[(0.075, 0.75)]
            elif dim == 1:
                bounds = [(0., 0.95)] #[(0.2, 0.75)]
            else:
                bounds = [(0., 0.9), (0., 0.9)]

            my_class = MlObj(model, all_params, all_targets, index, dim=dim)
            output = differential_evolution(my_class.objfn, bounds)
            return output.x

        resX = np.zeros((all_targets.shape[0],1))
        resY = np.zeros((all_targets.shape[0],1))

        for rep in range(repeats):
            results_x = Parallel(n_jobs=8)(delayed(recover)(all_params, all_targets, i, dim=0) for i in tqdm.tqdm(range(all_targets.shape[0]))) 
            results_y = Parallel(n_jobs=8)(delayed(recover)(all_params, all_targets, i, dim=1) for i in tqdm.tqdm(range(all_targets.shape[0]))) 

            resX += np.array(results_x)
            resY += np.array(results_y)

        resX /= repeats
        resY /= repeats

        for idx, (resx, resy) in enumerate(zip(resX, resY)):
            freq_rec.append(resx[0])
            amp_rec.append(resy[0])
            freq_true.append(all_params[idx,0].item())
            amp_true.append(all_params[idx,1].item())

        my_res = {'freq_true': freq_true, 'freq_rec': freq_rec, 'amp_true': amp_true, 'amp_rec': amp_rec}
        pickle.dump(my_res, open('MLEOutputV2.p','wb'))


    def inference(self, model, datamodule, stage, electrode_index, target_index, checkpointer=None, model_save_path=None):
        model = model.to(self.device)
        
        all_params = datamodule.test_dataset.tensors[0]
        all_targets = datamodule.test_dataset.tensors[1]
        print('Of {} available targets, selecting index {}'.format(all_targets.shape[0], target_index))
        
        # load a trained model if checkpointer or model_save_path is given
        if model_save_path is not None:
            checkpointer = fetch_last_checkpoint_model_filename(model_save_path)
        
        if checkpointer is not None:
            print(checkpointer)
            to_load = {"model": model}
            checkpoint = torch.load(checkpointer, map_location=self.device)
            Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

        target = all_targets[target_index] #.to(self.device)
        parameters = all_params[target_index]
        
        simulator = Simulator(model, num_electrodes=parameters.shape[0]-2, electrode_index=electrode_index, device=self.device)
        inference = Inference(
            num_rounds=self.num_rounds, 
            num_simulations=self.num_simulations, 
            simulation_batch_size=self.simulation_batch_size,
            training_batch_size=self.training_batch_size,
            num_samples=self.num_samples,
            filtering_ratio=self.filtering_ratio,
            num_proposals=self.num_proposals,
            timeout=self.timeout
        )

        if os.path.exists('density_estimator.pth'):
            print("Loading density estimator pth")
            posterior = torch.load('density_estimator.pth')
        else:
            import time
            st = time.time()
            # train NDE
            posterior = inference.train(simulator, target)
            et = time.time()
            print('Training time: {}'.format(et - st))

            if posterior is None:
                # finish the inference process when it is timeout
                return
            
            #torch.save(posterior, 'density_estimator.pth')

        import matplotlib.pyplot as plt

        # fancy plotting for the manuscript
        # this is for some null results!
        #stars = np.array([[0.684, 0.72], [0.1, 0.1]])
        stars = np.array([[0.99, 0.99]])

        '''
        test_params = parameters.unsqueeze(0).repeat(stars.shape[0], 1)
        test_params[:, 0] = torch.tensor(stars[:, 0])
        test_params[:, 1] = torch.tensor(stars[:, 1])
        test_params[:, 2+electrode_index] = 1.
        res, _ = model(test_params)
        plt.imshow(res.detach().numpy().T, vmin=0, vmax=1., cmap=plt.get_cmap('YlGn'))
        plt.axis('off')
        #plt.savefig('null.png')
        plt.show()
        '''

        path_x, path_y = inference.fancypairplot(posterior, target, parameters, stars=stars, contour=True)

        test_params = parameters.unsqueeze(0).repeat(path_x.shape[0],1)
        test_params[:, 0] = torch.tensor(path_y)
        test_params[:, 1] = torch.tensor(path_x)
        test_params[:, 2+electrode_index] = 1.
        res, _ = model(test_params)
        
        to_display = torch.cat([target.unsqueeze(0), res], axis=0)
        errs = torch.abs(to_display - target.unsqueeze(0)).sum(dim=1)
        to_display = to_display.T.detach().numpy()

        v_min, v_max = 0, to_display.max()

        fig = plt.figure()
        for k in range(to_display.shape[-1]):
            ax = fig.add_subplot(1, to_display.shape[-1], k+1)
            ax.imshow(np.expand_dims(to_display[:, k], -1), cmap=plt.cm.YlGn, vmin=v_min, vmax=v_max)
            ax.axis('off')
            ax.set_title('%0.2f'%(errs[k].item()))

        #plt.savefig('posterior_predictive.png', bbox_inches='tight')
        print(datamodule.selected_muscles)
        
        #plt.show(block=False)
        #import ipdb; ipdb.set_trace()
        
        st = time.time()
        # # plot posterior samples
        fig = inference.pairplot(posterior, target, parameters)
        et = time.time()
        print('Sampling Time: {}'.format(et -st))


        # fig.savefig('targetIdx%d_electrodeIdx%d.png' % (target_index, electrode_index))
        x, theta, log_probability, corr = inference.sampling_proposals(simulator, posterior, target)

        # save proposals as JSON file
        gt_ees = parameters.cpu().numpy()
        gt_electrode_index = np.nonzero(gt_ees[2:])[0][0]
        gt_electrode = datamodule.transformers['ees']['parameterization'].idx2elec[2**gt_electrode_index].item()
        gt_ees = datamodule.inverse_transform_ees(gt_ees[None,:2])

        x = x.cpu().numpy()
        theta = theta.cpu().numpy()
        proposed_electrode = datamodule.transformers['ees']['parameterization'].idx2elec[2**electrode_index].item()
        theta = datamodule.inverse_transform_ees(theta[:,:2])

        proposals = OrderedDict()
        proposals['gt_emg'] = target.cpu().numpy().tolist()
        proposals['gt_ees'] = {
            'electrode': gt_electrode,
            'freq': gt_ees['freq'][0].item(),
            'amp': gt_ees['amp'][0].item()
        }
        proposals['proposed_electrode'] = proposed_electrode
        for n in range(x.shape[0]):
            proposals['proposal_%d' % n] = {
                'emg': x[n].tolist(),            
                'ees': {
                    'freq': theta['freq'][n].item(),
                    'amp': theta['amp'][n].item()
                },           
                'log_probability': log_probability[n].item(),
                'corr': corr[n].item()
            }

        # write JSON
        with open('proposals.json', 'w', encoding="utf-8") as fp:
            json.dump(proposals, fp, ensure_ascii=False, indent="\t")
