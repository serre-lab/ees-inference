import torch
from torch import nn
import os
from ignite.engine import Events
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

@hydra.main(config_path="conf", config_name="config")
def run(cfg):
    # fix random seed
    torch.manual_seed(cfg.seed)
    
    model = instantiate(cfg.model)
    dm = instantiate(cfg.datamodule)
    trainer = instantiate(cfg.trainer)
    # prepare dataset
    dm.prepare_data()
    dm.setup(side=cfg.side) # select one fold over 10 folds

    if cfg.mode == 'eval':
        '''
        trainer.eval(
            model, 
            datamodule=dm,
            stage=cfg.stage, 
            checkpointer=cfg.checkpointer,
            model_save_path=cfg.model_save_path,
            visualization=cfg.visualization
        )
        '''
        trainer.get_metrics(
            model, 
            datamodule=dm,
            stage='test',
            electrode_index=cfg.electrode_index,
            target_index=cfg.target_index, 
            checkpointer=cfg.checkpointer,
            model_save_path=cfg.model_save_path
        )
    elif cfg.mode == 'MLE':
        trainer.MLE(
            model, 
            datamodule=dm,
            stage='test',
            checkpointer=cfg.checkpointer,
            model_save_path=cfg.model_save_path
        )
    elif cfg.mode == 'inference':
        trainer.inference(
            model, 
            datamodule=dm,
            stage='test',
            electrode_index=cfg.electrode_index,
            target_index=cfg.target_index, 
            checkpointer=cfg.checkpointer,
            model_save_path=cfg.model_save_path
        )
    else:
        trainer.train(model, datamodule=dm)
        if cfg.mode == 'train_then_eval':
            trainer.eval(
                model, 
                datamodule=dm,
                stage='test', 
                model_save_path=os.path.join(os.getcwd(), cfg.trainer.checkpoint_dir)
            )
            

if __name__ == "__main__":
    # parser = ArgumentParser()
    run()
