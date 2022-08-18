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

from param_recovery import LocalSimulator, GlobalSimulator, Inference
import json
from collections import OrderedDict

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

        # # attach running average to trainer
        # RunningAverage(output_transform=lambda x: x).attach(trainer, "running_avg_loss")
        # @trainer.on(Events.EPOCH_COMPLETED)
        # def log_running_avg_metrics(engine):
        #     print("Epoch[{}] Running avg loss: {:.2f}".format(engine.state.epoch, engine.state.metrics['running_avg_loss']))

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
            # 'optimizer': self.optimizer, 
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

        # 
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


    def inference(self, model, datamodule, stage, electrode_index, target_index, checkpointer=None, model_save_path=None):
        model = model.to(self.device)
        
        all_params = datamodule.test_dataset.tensors[0]
        all_targets = datamodule.test_dataset.tensors[1]
        elec_encoding = datamodule.elec_encoding
        print('Of {} available targets, selecting index {}'.format(all_targets.shape[0], target_index))
        
        # load a trained model if checkpointer or model_save_path is given
        if model_save_path is not None:
            checkpointer = fetch_last_checkpoint_model_filename(model_save_path)
        
        if checkpointer is not None:
            print(checkpointer)
            to_load = {"model": model}
            checkpoint = torch.load(checkpointer, map_location=self.device)
            Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

        target = all_targets[target_index]
        parameters = all_params[target_index]
        
        if elec_encoding == 'onehot':
            simulator = LocalSimulator(model, num_electrodes=parameters.shape[0]-2, electrode_index=electrode_index, device=self.device)
        else:
            simulator = GlobalSimulator(model, device=self.device)

        inference = Inference(
            elec_encoding=elec_encoding,
            num_rounds=self.num_rounds, 
            num_simulations=self.num_simulations, 
            simulation_batch_size=self.simulation_batch_size,
            training_batch_size=self.training_batch_size,
            num_samples=self.num_samples,
            filtering_ratio=self.filtering_ratio,
            num_proposals=self.num_proposals,
            timeout=self.timeout
        )
        # train NDE
        if elec_encoding == 'pos':
            pos = datamodule.xyi[:,:2]#torch.Tensor([*datamodule.xy2idx.keys()])
            # pos[:,1] = pos[:,1]# * 2.5   # TODO:
            width = datamodule.elec_size['w']
            height = datamodule.elec_size['h']# / 2
        else:
            pos = None
            width = height = None

        train = True
        # train = False
        if train:
            import time
            start = time.time()
            posterior = inference.train(simulator, target, _xy=pos, height=height, width=width)
            torch.save(posterior, 'global_model.pth')
            end = time.time()
            print('elapsed time: %.1f sec' % (end - start))
        else:
            posterior = torch.load('global_model.pth')#'inference.train(simulator, target)
            # posterior = torch.load('global_model_100K_5K.pth')#'inference.train(simulator, target)

        if posterior is None:
            # finish the inference process when it is timeout
            return

        # plot posterior samples
        fig = inference.pairplot(posterior, target, parameters)
        fig.savefig('targetIdx%d.png' % (target_index))

        x, theta, log_probability = inference.sampling_proposals(simulator, posterior, target)
        if elec_encoding == 'pos':
            # select proposals only when these are within an electrode
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle

            mask = None
            for c_x, c_y in pos:
                mask_x = torch.logical_and((theta[:,2] < c_x + width/2), (theta[:,2] > c_x - width/2))
                mask_y = torch.logical_and((theta[:,3] < c_y + height/2), (theta[:,3] > c_y - height/2))
                if mask is None:
                    mask = torch.logical_and(mask_x, mask_y)
                else:
                    mask = mask + torch.logical_and(mask_x, mask_y)

            fig = plt.figure()
            ax = fig.add_subplot()
            ax.scatter(theta[:,2], theta[:,3], marker='o', color='green', facecolors='None', alpha=0.1, s=1)
            ax.scatter(theta[mask][:,2], theta[mask][:,3], marker='o', color='blue', facecolors='None', alpha=0.2, s=1)
            ax.scatter(pos[:,0], pos[:,1], marker='o', color='red', s=10)

            # Add rectangles
            for c_x, c_y in pos:
                ax.add_patch(Rectangle(
                    xy=(c_x-width/2, c_y-height/2) ,width=width, height=height,
                    linewidth=1, color='blue', fill=False))
            # ax.axis('equal')
            plt.show()

            # ax.set_autoscale_on(False)
            ax.set_xlim((-1.5,1.5))
            # ax.set_ylim((-1,1))
            # ax.set_ylim((-2.5,2.5)) # TODO:
            fig.savefig('proposedElectrodes_targetIdx%d.png' % target_index)
            plt.close()

            if mask.sum() == 0:
                print('no proposals close to the electrodes')
                return

            x = x[mask]
            theta = theta[mask]
            log_probability = log_probability[mask]

        
        x, theta, log_probability, dist = inference.filtering_proposals(x, target, theta, log_probability, metric='l1')
        # x, theta, log_probability, dist = inference.filtering_proposals(x, target, theta, log_probability, metric='corr')

        def xy2idx(xy):
            x, y = xy
            # return x,y
            for c_x, c_y, idx in datamodule.xyi:
                if (x < c_x + width/2) and (x > c_x - width/2):
                    if (y < c_y + height/2) and (y > c_y - height/2):
                        return idx.item()

        # save proposals as JSON file
        gt_ees = parameters.cpu().numpy()
        gt_ees = datamodule.inverse_transform_ees(gt_ees[None,:])

        x = x.cpu().numpy()
        theta = theta.cpu().numpy()
        theta = datamodule.inverse_transform_ees(theta)
        if elec_encoding == 'onehot':
            proposed_electrode = datamodule.transformers['ees']['parameterization'].idx2elec[2**electrode_index].item()

        proposals = OrderedDict()
        proposals['gt_emg'] = target.cpu().numpy().tolist()
        proposals['gt_ees'] = {
            'freq': gt_ees['freq'][0].item(),
            'amp': gt_ees['amp'][0].item(),
            'elec': gt_ees['elec'][0].tolist() if elec_encoding=='onehot' else xy2idx(gt_ees['elec'][0].tolist())
        }
        
        for n in range(x.shape[0]):
            proposals['proposal_%d' % n] = {
                'emg': x[n].tolist(),            
                'ees': {
                    'freq': theta['freq'][n].item(),
                    'amp': theta['amp'][n].item(),
                    'elec': proposed_electrode if elec_encoding=='onehot' else xy2idx(theta['elec'][n].tolist())
                },
                'log_probability': log_probability[n].item(),
                'dist': dist[n].item()
            }

        # write JSON
        with open('proposals_targetIdx%d.json' % target_index, 'w', encoding="utf-8") as fp:
            json.dump(proposals, fp, ensure_ascii=False, indent="\t")
