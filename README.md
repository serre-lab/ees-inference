<p align="center">
<b> Fast Inference of Spinal Neuromodulation for Motor Control using Amortized Neural Networks </b>
<img src="thumbnail.png" width="640" height="480">
</p>
<p align="justify">
Epidural electrical stimulation (EES) has recently emerged as a potential therapeutic approach to restore motor function following chronic spinal cord injury (SCI). However, the lack of robust and systematic algorithms to automatically identify EES parameters that drive sensorimotor networks has become one of the main barriers to the clinical translation of EES. In this work, we present a novel, fully-automated computational framework to identify EES parameter combinations that optimally select for target muscle activation.
</p>

### Creating a runtime environment
### Datasets

### Forward model training
```
OMP_NUM_THREADS=1 python main.py hydra/launcher=joblib  \
                                 hydra.run.dir=<path to dir you want as a working directory> \
                                 model=mlp \
                                 mode=train \
                                 datamodule=sheep_20210610 \
                                 model_save_path=<path to save/load checkpoints> \
                                 model.network.out_size=7 \
                                 model.network.in_size=20 \
                                 trainer.device='cpu'

```
<p align="justify">
We note that model.network.out_size and model.network.in_size should be specified based on the number of EMG channels used and the EES parameter dimensionality respectively.
</p>


### Forward model evaluation
```
bash jobs/run.sh \
    --devce $device \
    --model $model \
    --mode "eval" \
    --dm $dm \
    --save_dir $save_dir \
    --side $side \
    --data_dir $data_dir \
    --out_size $out_size \
    --fold_idx $fold_idx \
    --working_dir $working_dir \
    --checkpointer_dir "models"
```

### Training the inverse model (Neural Density Estimator)
```
python main.py \
    hydra.run.dir=outputs/10-fold_cv_0/2020-12-08_20-04-17 \
    datamodule.data_dir=/Path/to/dataset/deep_spine/20200923 \
    model=mlp \
    mode=inference \
    model_save_path=/absolute/path/to/models/checkpoints_dir/ \
    electrode_index=0 \
    target_index=0
```
### Parallelize the training of electrode-conditioned inverse models
#### Requirements
```
# Install Hydra joblib plugin
pip install hydra-joblib-launcher --upgrade
```
#### Run
```
# examle of a multirun cmd, adding 1) hydra/launcher=joblib and 2) --multirun, and setting multiple values for electrode_index and target_index
python main.py \
    hydra/launcher=joblib \
    hydra.run.dir=outputs/10-fold_cv_0/2020-12-08_20-04-17 \
    datamodule.data_dir=/Path/to/dataset/deep_spine/20200923 \
    model=mlp \
    mode=inference \
    model_save_path=/absolute/path/to/models/checkpoints_dir/ \
    electrode_index=0,1,2,3,4,5 \
    target_index=0,1 \
    --multirun
```
