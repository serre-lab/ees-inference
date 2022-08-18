import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
import itertools
import os
import math
import json
import hashlib
import matplotlib.pyplot as plt
from utils.visualization import heatmap, annotate_heatmap

def read_dataset(data_dir, offset, window_size, selected_muscles):
    df = pd.read_parquet(os.path.join(data_dir, '_export_.parquet'), engine='pyarrow')
    # df = pd.read_parquet(os.path.join(data_dir, '_export_XY.parquet'), engine='pyarrow')
    
    stimulation_period = ['t=%.4f' % offset, 't=%.4f' % (offset + window_size)]
    
    electrodes = np.unique(df['StimElec'])
    channels = np.unique(df['Channel'])
    amplitudes = np.unique(df['StimAmp'])
    frequencies = np.unique(df['StimFreq'])

    # allocate lists to hold data from each trial

    configs = []
    for e in electrodes:
        e_mask = (df['StimElec'] == e)
        x, y = df.loc[e_mask]['StimX'].iloc[0].item(), df.loc[e_mask]['StimY'].iloc[0].item()
        for a in amplitudes:
            a_mask = (df['StimAmp'] == a)
            for f in frequencies:
                f_mask = (df['StimFreq'] == f)
                
                emg = []
                for i, m in enumerate(selected_muscles):
                    m_mask = (df['Channel'] == m + '_EMG')
                    samples = df.loc[e_mask & m_mask & a_mask & f_mask]
                    mean = samples.iloc[:, df.columns.get_loc('t=0.0000'):df.columns.get_loc(stimulation_period[0])].to_numpy().mean(1,keepdims=True)
                    samples = samples.iloc[:, df.columns.get_loc(stimulation_period[0]):df.columns.get_loc(stimulation_period[1])].to_numpy()
                    
                    # mean subtraction
                    emg.append(samples - mean)
                
                emg = np.stack(emg, axis=2) # repetition x time x muscle
                if emg.shape[0] > 0:
                    config = {
                        'ees': {
                            'freq': f,
                            'amp': a,
                            'elec': {
                                'idx': int(e),
                                'pos': (x, y)
                            },
                        },
                        'emg': emg.astype(np.float32)
                    }
                    configs.append(config)

    return configs


def cov_corr(x, y, axis):
    centered_x = x - x.mean(axis, keepdims=True)
    centered_y = y - y.mean(axis, keepdims=True)

    output_std = np.sqrt((centered_x ** 2).mean(axis))
    target_std = np.sqrt((centered_y ** 2).mean(axis))
    
    cov = (centered_x * centered_y).mean(axis)
    corr = cov / (output_std * target_std)
    
    return cov, corr


def split_half_reliability(
        configs,
        axis,
        num_iterations=100,
    ):

    train_emg = []
    valid_emg = []

    for conf in configs:
        emg = conf['emg']
        num_trials = emg.shape[0]
        for i in range(num_iterations):
            shuffled_indices = np.random.permutation(np.arange(num_trials))
            split = int(np.round(shuffled_indices.shape[0] * 0.5))

            train_indices = shuffled_indices[:split]
            valid_indices = shuffled_indices[split:]

            # averaged over repetitions
            train_emg.append(np.take(emg, train_indices, axis=0).mean(0))
            valid_emg.append(np.take(emg, valid_indices, axis=0).mean(0))

    train_emg = np.stack(train_emg, axis=0)
    valid_emg = np.stack(valid_emg, axis=0)
    train_emg = train_emg.reshape(len(configs), num_iterations, *train_emg.shape[1:])
    valid_emg = valid_emg.reshape(len(configs), num_iterations, *valid_emg.shape[1:])

    cov, corr = cov_corr(train_emg, valid_emg, axis=(axis+1))
    
    # average over iterations
    cov = cov.mean(1)
    corr = corr.mean(1)
    
    stats = {
        'cov': cov,
        'corr': corr
    }

    return stats


def filtering(
        configs,
        cov,        # configs x channel
        corr,       # configs x channel
        cov_threshold,
        corr_threshold, 
    ):

    # scaling covariance
    transformer = preprocessing.RobustScaler(quantile_range=(5, 95)).fit(cov)
    cov = (transformer.transform(cov))

    corr_mask = corr > corr_threshold     # filter unreliable EMG
    cov_mask = cov > cov_threshold        # filter subthreshold EMG
    # select configuration if at least one of the channels is above threshold
    mask = np.any(corr_mask * cov_mask, axis=1)

    print('noise ceiling before filtering: %.4f' % corr.mean())
    print('noise ceiling after filtering: %.4f' % corr[mask].mean())
    
    selected_configs = [conf for conf, m in zip(configs, mask) if m]

    return selected_configs


class Discretization():
    def __init__(self, q=99):
        self.q = q
        self.percentile = None

    def fit(self, x):
        # x: trial x time x channel
        rectified_x = np.abs(x)
        discretized_x = rectified_x.mean(1)     # trial x channel

        self.percentile = np.stack([np.percentile(discretized_x[:,i], self.q) for i in range(discretized_x.shape[-1])])
        # self.percentile = np.percentile(discretized_x.reshape(-1), self.q)

    def transform(self, x):
        assert self.percentile is not None

        # x: trial x time x channel
        rectified_x = np.abs(x)
        discretized_x = rectified_x.mean(1)     # trial x channel
        discretized_x = discretized_x / self.percentile[None,:]

        # clamps all elements in discretized_x to be smaller or equal 1
        # so that the range of discretized_x becomes [0, 1]
        discretized_x[discretized_x>1] = 1
        
        return discretized_x
        

class Parameterization():
    def __init__(self, elec_encoding):
        assert elec_encoding in ['onehot', 'pos']
        self.elec_encoding = elec_encoding

    def fit(self, x):
        # scaling freq and amp to [0-1]
        self.freq_scale = 1. / x['freq'].max()
        self.amp_scale = 1. / x['amp'].max()

        # scaling freq and amp to [0.-0.9]
        self.freq_scale = self.freq_scale * 0.9
        self.amp_scale = self.amp_scale * 0.9
        # self.freq_bias = 10
        self.freq_bias = 0.
        self.amp_bias = 0.

        if self.elec_encoding == 'onehot':
            self.elec2idx = {}
            self.idx2elec = {}
            electrodes = np.unique(x['elec']['idx'])
            self.num_electrodes = len(electrodes)
            for idx, e in enumerate(electrodes):
                # one_hot = np.zeros(self.num_electrodes)
                # one_hot[idx] = 1
                self.elec2idx[e] = 2**idx
                self.idx2elec[2**idx] = e

    def transform(self, x):
        assert self.freq_scale is not None
        assert self.amp_scale is not None

        freq = x['freq'][:,None]
        amp = x['amp'][:,None]

        if self.elec_encoding == 'onehot':
            # convert integer to binary vector
            elec = np.vectorize(self.elec2idx.get)(x['elec']['idx'])
            elec = ((elec[:,None] & (1 << np.arange(self.num_electrodes))) > 0).astype(int)
        else:
            elec = x['elec']['pos']

        parameterized_x = np.concatenate([freq, amp, elec], axis=1)
        # parameterized_x[:,0] = (parameterized_x[:,0] - self.freq_bias) * self.freq_scale
        parameterized_x[:,0] = (parameterized_x[:,0] * self.freq_scale) + self.freq_bias
        parameterized_x[:,1] = (parameterized_x[:,1] * self.amp_scale) + self.amp_bias

        return parameterized_x

    def inverse_transform(self, parameterized_x):
        # freq = (parameterized_x[:,0] * self.freq_scale) + self.freq_bias 
        freq = (parameterized_x[:,0] - self.freq_bias) / self.freq_scale
        amp = (parameterized_x[:,1] - self.amp_bias) / self.amp_scale

        if parameterized_x.shape[1] == 2:
            return {
                'freq': freq,
                'amp': amp
            }

        if self.elec_encoding == 'onehot':
            assert parameterized_x[:,2:].shape[1] == self.num_electrodes

            # convert binary vector to integer
            elec = (parameterized_x[:,2:] * (1 << np.arange(self.num_electrodes))).sum(1).astype(int)
            elec = np.vectorize(self.idx2elec.get)(elec)

            return {
                'freq': freq,
                'amp': amp,
                'elec': elec
            }
        else:
            return {
                'freq': freq,
                'amp': amp,
                'elec': parameterized_x[:,2:]
            }

def preprocess(train_data, test_data, preprocess_cfg):
    # preprocess EMG
    ## dicretization
    discretization = Discretization(**preprocess_cfg['emg'])
    discretization.fit(train_data['emg'])
    train_data['transformed_emg'] = discretization.transform(train_data['emg'])
    test_data['transformed_emg'] = discretization.transform(test_data['emg'])

    # preprocess EES
    ## parameterization
    # num_electrodes = train_data['ees'].shape[-1]
    parameterization = Parameterization(**preprocess_cfg['ees'])
    parameterization.fit(train_data['ees'])
    
    train_data['transformed_ees'] = parameterization.transform(train_data['ees'])
    test_data['transformed_ees'] = parameterization.transform(test_data['ees'])
    
    return {
        'emg': {
            'discretization': discretization
        },
        'ees': {
            'parameterization': parameterization,
            # 'scaler': scaler
        }
    }


def train_test_split(configs, holdout_frequencies, holdout_amplitudes):
    train_configs = []
    test_configs = []
    # for conf in configs:
    #     # if (conf['ees']['elec']['idx'] != 1) and (conf['ees']['elec']['idx'] < 129):  #TODO: dirty fix only for exp20210604  
    #     if (conf['ees']['elec']['idx'] != 1):  #TODO: dirty fix only for exp20210604  
    #         if conf['ees']['freq'] in holdout_frequencies or conf['ees']['amp'] in holdout_amplitudes:
    #             test_configs.append(conf)
    #         else:
    #             train_configs.append(conf)

    for conf in configs:
        if conf['ees']['freq'] in holdout_frequencies or conf['ees']['amp'] in holdout_amplitudes:
            test_configs.append(conf)
        else:
            train_configs.append(conf)

    
    # train_data = {
    #     'ees': {
    #         'freq': np.array([conf['ees']['freq'] for conf in train_configs for _ in range(conf['emg'].shape[0])], dtype=np.float32),
    #         'amp': np.array([conf['ees']['amp'] for conf in train_configs for _ in range(conf['emg'].shape[0])], dtype=np.float32),
    #         'elec': np.array([conf['ees']['elec'] for conf in train_configs for _ in range(conf['emg'].shape[0])], dtype=np.int32),
    #     },
    #     'emg': np.concatenate([conf['emg'] for conf in train_configs], axis=0),
    #     'config': np.array([idx for idx, conf in enumerate(train_configs) for _ in range(conf['emg'].shape[0])], dtype=np.int32)
    # }
    # test_data = {
    #     'ees': {
    #         'freq': np.array([conf['ees']['freq'] for conf in test_configs for _ in range(conf['emg'].shape[0])], dtype=np.float32),
    #         'amp': np.array([conf['ees']['amp'] for conf in test_configs for _ in range(conf['emg'].shape[0])], dtype=np.float32),
    #         'elec': np.array([conf['ees']['elec'] for conf in test_configs for _ in range(conf['emg'].shape[0])], dtype=np.int32),
    #     },
    #     'emg': np.concatenate([conf['emg'] for conf in test_configs], axis=0),
    #     'config': np.array([idx for idx, conf in enumerate(test_configs) for _ in range(conf['emg'].shape[0])], dtype=np.int32)
    # }
    

    train_data = {
        'ees': {
            'freq': np.array([conf['ees']['freq'] for conf in train_configs], dtype=np.float32),
            'amp': np.array([conf['ees']['amp'] for conf in train_configs], dtype=np.float32),
            'elec': {
                'idx': np.array([conf['ees']['elec']['idx'] for conf in train_configs], dtype=np.int32),
                'pos': np.array([conf['ees']['elec']['pos'] for conf in train_configs], dtype=np.float32)
            },
        },
        'emg': np.concatenate([conf['emg'].mean(0, keepdims=True) for conf in train_configs], axis=0),
        'config': np.array([idx for idx, conf in enumerate(train_configs)], dtype=np.int32)
    }
    test_data = {
        'ees': {
            'freq': np.array([conf['ees']['freq'] for conf in test_configs], dtype=np.float32),
            'amp': np.array([conf['ees']['amp'] for conf in test_configs], dtype=np.float32),
            'elec': {
                'idx': np.array([conf['ees']['elec']['idx'] for conf in test_configs], dtype=np.int32),
                'pos': np.array([conf['ees']['elec']['pos'] for conf in test_configs], dtype=np.float32)
            },
        },
        'emg': np.concatenate([conf['emg'].mean(0, keepdims=True) for conf in test_configs], axis=0),
        'config': np.array([idx for idx, conf in enumerate(test_configs)], dtype=np.int32)
    }
    # import pdb
    # pdb.set_trace()
    
    return train_data, test_data

class SheepDataModule():
    def __init__(self, 
        data_dir,
        offset,
        window_size,
        selected_muscles,
        elec_encoding,
        q,
        holdout_frequencies,
        holdout_amplitudes,
        train_bs,
        valid_bs,
        test_bs,
        seed=1988,
        device='cpu'
    ):
        self.data_dir = data_dir.rstrip('//')
        self.offset = offset
        self.window_size = window_size
        self.selected_muscles = selected_muscles
        self.elec_encoding = elec_encoding
        self.q = q
        self.holdout_frequencies = holdout_frequencies
        self.holdout_amplitudes = holdout_amplitudes
        self.train_bs = train_bs
        self.valid_bs = valid_bs
        self.test_bs = test_bs
        self.seed = seed
        self.device = device
        
        # fix random seed
        np.random.seed(self.seed)

    def prepare_data(self):
        _dir = os.path.join(self.data_dir, self.elec_encoding)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        save_path = os.path.join(_dir, 'data.npy')

        if os.path.isfile(save_path):
            data = np.load(save_path, allow_pickle=True).item()
        else:
            configs = read_dataset(
                self.data_dir, 
                offset=self.offset, 
                window_size=self.window_size, 
                selected_muscles=self.selected_muscles
            )

            train_data, test_data = train_test_split(
                configs,
                holdout_frequencies=self.holdout_frequencies,
                holdout_amplitudes=self.holdout_amplitudes
            )

            transformers = preprocess(
                train_data, 
                test_data, 
                preprocess_cfg={
                    'ees': {
                        'elec_encoding': self.elec_encoding
                    },
                    'emg': {
                        'q': self.q
                    }
            })

            # save test EMG figure
            def visualize_dataset(file, data):
                transformed_emg =[]
                legend = []
                for c in np.unique(data['config']):
                    freq = data['ees']['freq'][data['config'] == c][0]
                    amp = data['ees']['amp'][data['config'] == c][0]
                    elec = data['ees']['elec']['idx'][data['config'] == c][0]
                    transformed_emg.append(data['transformed_emg'][data['config'] == c].mean(0))

                    if transformers['ees']['parameterization'].elec_encoding == 'onehot':
                        idx = transformers['ees']['parameterization'].elec2idx[elec]
                        legend.append('[%d]: %.0fHz, %.0fuA, %d(elec)/%d(idx)' % (c, freq, amp, elec, int(math.log(idx, 2))))
                    else:
                        x, y = data['ees']['elec']['pos'][data['config'] == c][0]
                        legend.append('[%d]: %.0fHz, %.0fuA, %d(elec)/[%.1f, %.1f](pos)' % (c, freq, amp, elec, x, y))

                transformed_emg = np.stack(transformed_emg, axis=0)
                fig, ax = plt.subplots(figsize=(transformed_emg.shape[1],transformed_emg.shape[0]))
                im, cbar = heatmap(transformed_emg, legend, self.selected_muscles, ax=ax, vmin=0, vmax=0.9, cmap="YlGn", cbarlabel="muscle recruitment")
                texts = annotate_heatmap(im, valfmt="{x:.2f}")
                cbar.remove()
                fig.tight_layout()
                plt.savefig(file)
                plt.close()
            
            visualize_dataset(os.path.join(self.data_dir, self.elec_encoding, 'train_dataset.png'), train_data)
            visualize_dataset(os.path.join(self.data_dir, self.elec_encoding, 'test_dataset.png'), test_data)

            import pdb
            pdb.set_trace()
            np.save(save_path, {
                'train_data': train_data,
                'test_data': test_data,
                'transformers': transformers
            })

        electrodes = np.unique(data['train_data']['ees']['elec']['idx'])
        with open(os.path.join(self.data_dir, 'Louise-electrodeMap.json'), 'r') as f:   #TODO:
            data = json.load(f)
            self.elec_size = {
                'h': data['elec_height_y'],
                'w': data['elec_width_x']
            }

            xyi = []
            for i, e in enumerate(data['Electrode']):
                if e in electrodes: #TODO: don't use the electrodes that are not used during training for now
                    x = data['X'][i]
                    y = data['Y'][i]
                    xyi.append([x,y,e])
            self.xyi = torch.Tensor(xyi)

        
    def setup(self, side='both'):
        assert side in ['left', 'right', 'both']

        data = np.load(os.path.join(self.data_dir, self.elec_encoding, 'data.npy'), allow_pickle=True).item()
        
        train_data = data['train_data']
        test_data = data['test_data']

        # select left or right side EMG channels
        if side in ['left', 'right']:
            s = 'L' if side == 'left' else 'R'
            train_data['transformed_emg'] = train_data['transformed_emg'][:,[ch[0]==s for ch in self.selected_muscles]]
            test_data['transformed_emg'] = test_data['transformed_emg'][...,[ch[0]==s for ch in self.selected_muscles]]
        
        # numpy array to torch tensor
        train_ees = torch.Tensor(train_data['transformed_ees'])
        train_emg = torch.Tensor(train_data['transformed_emg'])
        test_ees = torch.Tensor(test_data['transformed_ees'])
        test_emg = torch.Tensor(test_data['transformed_emg'])

        # device conversion
        train_ees  = train_ees.to(self.device)
        train_emg  = train_emg.to(self.device)
        test_ees  = test_ees.to(self.device)
        test_emg  = test_emg.to(self.device)

        self.train_dataset = TensorDataset(train_ees, train_emg)
        self.valid_dataset = TensorDataset(test_ees, test_emg)
        self.test_dataset = TensorDataset(test_ees, test_emg)

        # transformers used for preprocessing EMG and EES
        self.transformers = data['transformers']
        
    def inverse_transform_ees(self, X):
        x = X.copy()

        parameterization = self.transformers['ees']['parameterization']
        # scaler = self.transformers['ees']['scaler']

        # x[:,:2] = scaler.inverse_transform(x[:,:2])
        x = parameterization.inverse_transform(x)
        
        return x

    def train_dataloader(self):
        train_bs = len(self.train_dataset) if self.train_bs is None else self.train_bs
        return DataLoader(self.train_dataset, batch_size=train_bs, shuffle=True, num_workers=0, drop_last=True)

    def valid_dataloader(self):
        valid_bs = len(self.valid_dataset) if self.valid_bs is None else self.valid_bs
        return DataLoader(self.valid_dataset, batch_size=valid_bs, shuffle=False, num_workers=0, drop_last=False)

    def test_dataloader(self):
        test_bs = len(self.test_dataset) if self.test_bs is None else self.test_bs
        return DataLoader(self.test_dataset, batch_size=test_bs, shuffle=False, num_workers=0, drop_last=False)


# if __name__ == "__main__":
#     dm = SheepDataModule(
#         data_dir='/media/data_cifs_lrs/projects/prj_deepspine/data/SheepData/20200923/',
#         offset=100,
#         window_size=300,
#         selected_muscles=['RThoracolumbarFascia', 'RGracilis', 'RTensorFasciaeLatae', 'RPeroneusLongus', 'RBicepsFemoris'],
#         cov_threshold=0.1,
#         corr_threshold=0.4,
#         num_folds=10,
#         train_bs=None,
#         valid_bs=None,
#         test_bs=None,
#     )
    
#     start = time.time()
#     dm.prepare_data()
#     dm.setup(1)
#     end = time.time()
#     print('elapsed time: %f' % (end - start))