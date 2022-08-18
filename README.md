# Deep Spine
## Datasets
### Sheep Dataset
* 20200923 dataset: https://drive.google.com/file/d/15rsHn90EPBDH_uBBA_CmglhFxdf8Skqy/view?usp=sharing
* 20201019 dataset: https://drive.google.com/file/d/10WBeVNth93qDANM_B_1SbV2c1iyIr6GY/view?usp=sharing
## Training
```
bash jobs/run.sh \
    --devce $device \
    --model $model \
    --mode "train_then_eval" \
    --dm $dm \
    --save_dir $save_dir \
    --side $side \
    --data_dir $data_dir \
    --out_size $out_size
```
## Evaluation
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

## NDE training
### Single run 
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
### Multi-run with Joblib
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

