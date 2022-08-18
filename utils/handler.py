import numpy as np
import os
from ignite.utils import convert_tensor

def output_transform(output):
    y_pred = output['y_pred']
    y = output['y']
    return y_pred, y

def prepare_batch(batch, device, non_blocking=False):
    """Prepare batch for training: pass to a device with options.
    """
    x, y = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )

def switch_batch(engine, device):
    engine.state.batch = prepare_batch(engine.state.batch, device)

# loading the saved model
def fetch_last_checkpoint_model_filename(model_save_path):
    checkpoint_files = os.listdir(model_save_path)
    checkpoint_files = [f for f in checkpoint_files if '.pt' in f]
    checkpoint_iter = [
        int(x.split('_')[1])
        for x in checkpoint_files]
    last_idx = np.array(checkpoint_iter).argmax()
    return os.path.join(model_save_path, checkpoint_files[last_idx])

def save_activation(engine, activation_dict):
    for key in engine.state.output.keys():
        activation_dict[key].append(engine.state.output[key])
    