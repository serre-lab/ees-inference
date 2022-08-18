import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
import itertools  
import os
import json
import hashlib
import matplotlib.pyplot as plt
from utils.visualization import heatmap, annotate_heatmap

def read_datasetV0(
        data_dir,
        offset,
        window_size,
        selected_muscles
    ):
    cropEdgesTimes = [-100e-3, 400e-3]

    # allocate lists to hold data from each trial
    eesList = []
    emgList = []
    
    with pd.HDFStore(os.path.join(data_dir, '_emg_XS_export.h5'), 'r') as store:
        # each trial has its own eesKey, get list of all
        allEESKeys = [
            i
            for i in store.keys()
            if ('stim' in i)]
        
        for idx, eesKey in enumerate(sorted(allEESKeys)):
            # data for this trial is stored in a pd.dataframe
            stimData = pd.read_hdf(store, eesKey)
            # metadata is stored in a dictionary
            eesMetadata = store.get_storer(eesKey).attrs.metadata
            # extract column names from first trial
            if idx == 0:
                eesColumns = [cn[0] for cn in stimData.columns if cn[1] == 'amplitude']
                emgColumns = [cn[0] for cn in stimData.columns if cn[1] == 'emg_env']
                metadataColumns = sorted([k for k in eesMetadata.keys()])
                eesColIdx = [cn for cn in stimData.columns if cn[1] == 'amplitude']
                emgColIdx = [cn for cn in stimData.columns if cn[1] == 'emg_env']
                metadataDF = pd.DataFrame(
                    None, index=range(len(allEESKeys)),
                    columns=metadataColumns)
            metadataDF.loc[idx, :] = eesMetadata
            # get mask for requested time points
            cropEdgesMask = (
                (stimData.index >= cropEdgesTimes[0]) &
                (stimData.index <= cropEdgesTimes[1]))
            eesList.append(stimData.loc[cropEdgesMask, eesColIdx])
            emgList.append(stimData.loc[cropEdgesMask, emgColIdx])

    # metadataNP.shape = trials x metadata type
    # metadata column names are in metadataColumns
    # globalIdx is the index of the trial
    # combinationIdx is the index of the particular combination
    # of rate, active electrodes and amplitude
    with pd.HDFStore(os.path.join(data_dir, '_emg_XS_export.h5'), 'r') as store:
        noiseCeilDF = pd.read_hdf(store, 'noiseCeil').unstack(level='feature')
        noiseCeilDF.index.set_names('amplitude', level='nominalCurrent', inplace=True)
        columnLabels = noiseCeilDF.columns.to_list()
        electrodeLabels = noiseCeilDF.index.get_level_values('electrode').to_list()
        amplitudeLabels = noiseCeilDF.index.get_level_values('amplitude').to_list()
        covariances = pd.read_hdf(store, 'covariance').unstack(level='feature').to_numpy()
        noiseCeil = noiseCeilDF.to_numpy()

    # add the configuration indices into metadata
    noiseCeilMeta = noiseCeilDF.index.to_frame(index=False)
    def getEESIdx(metaRow):
        findMask = (noiseCeilMeta['electrode'] == metaRow['electrode']) & (noiseCeilMeta['RateInHz'] == metaRow['RateInHz']) & (noiseCeilMeta['amplitude'] == metaRow['amplitude'])
        if not noiseCeilMeta.index[findMask].empty:
            return noiseCeilMeta.index[findMask][0]
        else:
            return np.nan
    metadataDF['eesIdx'] = metadataDF.apply(getEESIdx, axis=1)

    
    # convert to numpy
    eesNP = np.stack(eesList)   # trial x time x electrode
    emgNP = np.stack(emgList)   # trial x time x channel
    metadataNP = metadataDF.to_numpy()

    # remove outliers
    outliers = metadataDF.loc[:,'outlierTrial'].to_numpy().astype('bool')
    eesNP = eesNP[~outliers]
    emgNP = emgNP[~outliers]
    metadataNP = metadataNP[~outliers]

    # temporal crop
    eesNP = eesNP[:, offset:offset+window_size]
    emgNP = emgNP[:, offset:offset+window_size]
    
    # select muscles
    if selected_muscles is not None:
        emgNP = emgNP[..., [muscle in selected_muscles for muscle in emgColumns]]
        emgColumns = [muscle for muscle in emgColumns if muscle in selected_muscles]

    # add the activated electrode location into meta
    binarized_ees = np.abs(np.sign(eesNP.sum(1)))
    num_electrodes = binarized_ees.shape[1]
    loc = (binarized_ees * (1 << np.arange(num_electrodes))).sum(1).astype(int)

    # call copy() to make numpy array contiguous 
    data = {
        'ees': eesNP.copy().astype(np.float32),
        'emg': emgNP.copy().astype(np.float32),
        'meta': {
            'freq': metadataNP[:,0].astype(np.float32),
            'amp': metadataNP[:,1].astype(np.float32),
            'config': metadataNP[:,-1].astype(int),
            'loc': loc
        },
        'eesColumns': np.array(eesColumns),
        'emgColumns': np.array(emgColumns),
        'metaColumns': metadataDF.columns.to_numpy().astype('U')
    }
    return data


def cov_corrV0(x, y, axis):
    centered_x = x - x.mean(axis, keepdims=True)
    centered_y = y - y.mean(axis, keepdims=True)

    output_std = np.sqrt((centered_x ** 2).mean(axis))
    target_std = np.sqrt((centered_y ** 2).mean(axis))
    
    cov = (centered_x * centered_y).mean(axis)
    corr = cov / (output_std * target_std)
    
    return cov, corr


def split_half_reliabilityV0(
        data,
        axis,
        num_iterations=100,
        return_splits=False
    ):
    emg = data['emg']
    meta = data['meta']

    ees_configs = np.unique(meta['config'])

    if return_splits:
        ees = data['ees']
        train_ees = []
        valid_ees = []
    train_emg = []
    valid_emg = []

    for c in ees_configs:
        selected_indices = np.argwhere(meta['config']==c).squeeze()
        for i in range(num_iterations):
            shuffled_indices = np.random.permutation(selected_indices)
            split = int(np.round(shuffled_indices.shape[0] * 0.5))

            train_indices = shuffled_indices[:split]
            valid_indices = shuffled_indices[split:]

            if return_splits:
                # take the first sample because all ees samples are same over repetitions
                train_ees.append(np.take(ees, train_indices, axis=0)[0])
                valid_ees.append(np.take(ees, valid_indices, axis=0)[0])
            # averaged over repetitions
            train_emg.append(np.take(emg, train_indices, axis=0).mean(0))
            valid_emg.append(np.take(emg, valid_indices, axis=0).mean(0))

    train_emg = np.stack(train_emg, axis=0)
    valid_emg = np.stack(valid_emg, axis=0)
    train_emg = train_emg.reshape(ees_configs.size, num_iterations, *train_emg.shape[1:])
    valid_emg = valid_emg.reshape(ees_configs.size, num_iterations, *valid_emg.shape[1:])

    cov, corr = cov_corrV0(train_emg, valid_emg, axis=(axis+1))
    # average over iterations
    cov = cov.mean(1)
    corr = corr.mean(1)
    stats = {
        'cov': cov,
        'corr': corr
    }

    if return_splits:
        train_ees = np.stack(train_ees, axis=0)
        valid_ees = np.stack(valid_ees, axis=0)
        train_ees = train_ees.reshape(ees_configs.size, num_iterations, *train_ees.shape[1:])
        valid_ees = valid_ees.reshape(ees_configs.size, num_iterations, *valid_ees.shape[1:])
    
        splits = {
            'train_ees': train_ees,
            'valid_ees': valid_ees,
            'train_emg': train_emg,
            'valid_emg': valid_emg
        }
        return stats, splits
    
    return stats

def filteringV0(
        data,
        cov,        # configs x channel
        corr,       # configs x channel
        cov_threshold,
        corr_threshold, 
        save_path
    ):
    ees = data['ees']       # trial x time x electrode
    emg = data['emg']       # trial x time x channel
    meta = data['meta']     

    ees_configs = np.unique(meta['config'])
    
    # scaling covariance
    transformer = preprocessing.RobustScaler(quantile_range=(5, 95)).fit(cov)
    cov = (transformer.transform(cov))

    corr_mask = corr > corr_threshold     # filter unreliable EMG
    cov_mask = cov > cov_threshold        # filter subthreshold EMG
    # select configuration if at least one of the channels is above threshold
    mask = np.any(corr_mask * cov_mask, axis=1)
    selected_ees_configs = ees_configs[mask]

    print('noise ceiling before filtering: %.4f' % corr.mean())
    print('noise ceiling after filtering: %.4f' % corr[mask].mean())
    
    selected_indices = [meta['config']==c for c in selected_ees_configs]
    selected_indices = np.stack(selected_indices)
    selected_indices = np.any(selected_indices, axis=0)
    
    data['ees'] = ees[selected_indices]
    data['emg'] = emg[selected_indices]
    data['meta'] = {k:v[selected_indices] for (k, v) in meta.items()}

    # save
    np.save(save_path, data)


class DiscretizationV0():
    def __init__(self, q=99):
        self.q = q
        self.percentile = None

    def fit(self, x):
        # x: trial x time x channel
        rectified_x = np.abs(x)
        discretized_x = rectified_x.mean(1)     # trial x channel

        self.percentile = np.stack([np.percentile(discretized_x[:,i], self.q) for i in range(discretized_x.shape[-1])])

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
        

class ParameterizationV0():
    def __init__(self, num_electrodes):
        self.freq_scale = None
        self.amp_scale = None
        self.num_electrodes = num_electrodes

    def fit(self, x):
        assert x['freq'].min() > 0, "freq should be positive"
        # assume the dataset has only negative amplitudes
        assert x['amp'].max() < 0, "amp should be negative"

        # scaling freq and amp to [0-1]
        self.freq_scale = 1. / x['freq'].max()
        self.amp_scale = 1. / x['amp'].min()

        # scaling freq and amp to [0.-0.9]
        self.freq_scale = self.freq_scale * 0.9
        self.amp_scale = self.amp_scale * 0.9
        self.freq_bias = 0.
        self.amp_bias = 0.

    def transform(self, x):
        assert self.freq_scale is not None
        assert self.amp_scale is not None

        freq = x['freq'][:,None]
        amp = x['amp'][:,None]
        # convert integer to binary vector
        loc = ((x['loc'][:,None].astype(int) & (1 << np.arange(self.num_electrodes))) > 0).astype(int)
        
        parameterized_x = np.concatenate([freq, amp, loc], axis=1)
        parameterized_x[:,0] = (parameterized_x[:,0] * self.freq_scale) + self.freq_bias
        parameterized_x[:,1] = (parameterized_x[:,1] * self.amp_scale) + self.amp_bias

        return parameterized_x

    def inverse_transform(self, parameterized_x):
        freq = (parameterized_x[:,0] - self.freq_bias)/ self.freq_scale
        amp = (parameterized_x[:,1] - self.amp_bias) / self.amp_scale

        if parameterized_x.shape[1] == 2:
            return {
                'freq': freq,
                'amp': amp
            }
        
        assert parameterized_x[:,2:].shape[1] == self.num_electrodes

        # convert binary vector to integer
        loc = (parameterized_x[:,2:] * (1 << np.arange(self.num_electrodes))).sum(1).astype(int)

        return {
            'freq': freq,
            'amp': amp,
            'loc': loc
        }


def preprocessV0(train_data, test_data, preprocess_cfg):
    # preprocess EMG
    ## dicretization
    discretization = DiscretizationV0(**preprocess_cfg['emg'])
    discretization.fit(train_data['emg'])
    train_data['emg'] = discretization.transform(train_data['emg'])
    test_data['emg'] = discretization.transform(test_data['emg'])

    # preprocess EES
    ## parameterization
    num_electrodes = train_data['ees'].shape[-1]
    parameterization = ParameterizationV0(num_electrodes)
    parameterization.fit(train_data['meta'])
    
    train_data['ees'] = parameterization.transform(train_data['meta'])
    test_data['ees'] = parameterization.transform(test_data['meta'])
    
    return {
        'emg': {
            'discretization': discretization
        },
        'ees': {
            'parameterization': parameterization,
            # 'scaler': scaler
        }
    }


def k_fold_cv(data, num_folds, seed, data_dir, fold_names, preprocess_cfg):
    ees = data['ees']       # trial x time x electrode
    emg = data['emg']       # trial x time x channel
    meta = data['meta']     

    # K fold cross validation
    ees_configs = np.unique(meta['config'])
    kf=KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    
    for fold_idx, (train_set, test_set) in enumerate(kf.split(ees_configs)):
        save_path = os.path.join(data_dir, fold_names[fold_idx])
        if not os.path.isfile(save_path):
            train_ees_configs = ees_configs[train_set]
            test_ees_configs = ees_configs[test_set]
            
            
            train_indices = [meta['config']==c for c in train_ees_configs]
            train_indices = np.stack(train_indices)
            train_indices = np.any(train_indices, axis=0)
            test_indices = [meta['config']==c for c in test_ees_configs]
            test_indices = np.stack(test_indices)
            test_indices = np.any(test_indices, axis=0)

            # train
            train_data = {
                'ees': ees[train_indices],
                'emg': emg[train_indices],
                'meta': {k:v[train_indices] for (k, v) in meta.items()}
            }
            # test
            test_data = {
                'ees': ees[test_indices],
                'emg': emg[test_indices],
                'meta': {k:v[test_indices] for (k, v) in meta.items()}
            }
            
            # preprocess EMG & EES
            transformers = preprocessV0(train_data, test_data, preprocess_cfg)

            # compute noise ceiling for test set
            # get the splits used for noise ceiling computation
            test_stats, test_splits = split_half_reliabilityV0(test_data, axis=1, return_splits=True)
            corr = test_stats['corr'].mean()
            print('noise ceiling of test fold %d: %.4f' % (fold_idx, corr))

            # save
            np.save(save_path, {
                'train_data': train_data,
                'test_data': test_data,
                'test_splits': test_splits,
                'corr': corr,
                'transformers': transformers
            })

class SheepDataModuleV0():
    def __init__(self, 
        data_dir,
        offset,
        window_size,
        selected_muscles,
        cov_threshold,
        corr_threshold,
        q,
        num_folds,
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
        self.cov_threshold = cov_threshold
        self.corr_threshold = corr_threshold
        self.q = q
        self.num_folds = num_folds
        self.train_bs = train_bs
        self.valid_bs = valid_bs
        self.test_bs = test_bs
        self.seed = seed
        self.device = device
        
        # fix random seed
        np.random.seed(self.seed)
        
        prepare_cfg = {
            'data_dir': self.data_dir,
            'offset': self.offset,
            'window_size': self.window_size,
            'selected_muscles': [*self.selected_muscles],    # convert hydra ListConfig to list
            'cov_threshold': self.cov_threshold,
            'corr_threshold': self.corr_threshold,
            'seed': self.seed
        }
        self.name = hashlib.md5(json.dumps(prepare_cfg, sort_keys=True).encode()).hexdigest() + '.npy'

        prepare_cfg['num_folds'] = self.num_folds
        prepare_cfg['q'] = self.q
        self.fold_names = []
        for fold_idx in range(self.num_folds):
            prepare_cfg['fold_idx'] = fold_idx
            self.fold_names.append(hashlib.md5(json.dumps(prepare_cfg, sort_keys=True).encode()).hexdigest() + '.npy')
        

    def prepare_data(self):
        save_path = os.path.join(self.data_dir, self.name)
        if os.path.isfile(save_path):
            data = np.load(save_path, allow_pickle=True).item()
        else:
            data = read_datasetV0(
                self.data_dir, 
                offset=self.offset, 
                window_size=self.window_size, 
                selected_muscles=self.selected_muscles
            )
            stats = split_half_reliabilityV0(data, axis=1)
            filteringV0(
                data,
                stats['cov'],
                stats['corr'],
                cov_threshold=self.cov_threshold,
                corr_threshold=self.corr_threshold, 
                save_path=save_path
            )

        self.ees_channels = data['eesColumns']
        self.emg_channels = data['emgColumns']

        k_fold_cv(
            data, 
            num_folds=self.num_folds,
            seed=self.seed,
            data_dir=self.data_dir,
            fold_names=self.fold_names,
            preprocess_cfg={
                'emg': {
                    'q': self.q
                }
            }
        )
        

    def setup(self, fold_idx, side='both'):
        assert fold_idx in list(range(self.num_folds))
        assert side in ['left', 'right', 'both']

        data = np.load(os.path.join(self.data_dir, self.fold_names[fold_idx]), allow_pickle=True).item()
        train_data = data['train_data']
        test_splits = data['test_splits']
        print('noise ceiling of test fold %d: %.4f' % (fold_idx, data['corr']))
        
        # select left or right side EMG channels
        if side in ['left', 'right']:
            s = 'L' if side == 'left' else 'R'
            train_data['emg'] = train_data['emg'][:,[ch[0]==s for ch in self.emg_channels]]
            test_splits['valid_emg'] = test_splits['valid_emg'][...,[ch[0]==s for ch in self.emg_channels]]
        
        # numpy array to torch tensor
        train_ees = torch.Tensor(train_data['ees'])
        train_emg = torch.Tensor(train_data['emg'])
        test_ees = torch.Tensor(test_splits['valid_ees'])
        test_emg = torch.Tensor(test_splits['valid_emg'])
        valid_ees = test_ees.view(-1, test_ees.shape[-1])
        valid_emg = test_emg.view(-1, test_emg.shape[-1])
        #TODO: implement a custom test dataset to support getting average over repetitions for stim estimation
        ## current version is the average over splits to reuse these
        test_ees = test_ees.mean(1)
        test_emg = test_emg.mean(1)
        
        # device conversion
        train_ees  = train_ees.to(self.device)
        train_emg  = train_emg.to(self.device)
        valid_ees  = valid_ees.to(self.device)
        valid_emg  = valid_emg.to(self.device)
        test_ees  = test_ees.to(self.device)
        test_emg  = test_emg.to(self.device)

        
        self.train_dataset = TensorDataset(train_ees, train_emg)
        self.valid_dataset = TensorDataset(valid_ees, valid_emg)
        self.test_dataset = TensorDataset(test_ees, test_emg)

        # transformers used for preprocessing EMG and EES
        self.transformers = data['transformers']
        
        # save test EMG figure
        legends = []
        stimulations = test_ees.cpu().numpy().copy()
        theta = self.inverse_transform_ees(stimulations[:,:2])
        electrodes = stimulations[:,2:]
        for n in range(test_emg.shape[0]):
            freq = theta['freq'][n]
            amp = theta['amp'][n]
            electrode = self.ees_channels[electrodes[n]==1].item()

            legends.append('%.1fHz, %.0fuA, %s' % (freq, amp, electrode))
            
        fig, ax = plt.subplots(figsize=(test_emg.shape[1],test_emg.shape[0]))
        im, cbar = heatmap(test_emg, legends, self.emg_channels.tolist(), ax=ax, vmin=0, vmax=0.9, cmap="YlGn", cbarlabel="muscle recruitment")
        texts = annotate_heatmap(im, valfmt="{x:.2f}")
        cbar.remove()
        fig.tight_layout()#(rect=[0, 0.03, 1, 0.95])
        plt.savefig('vis_testDataset_fold_%d.png' % fold_idx)
        
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
