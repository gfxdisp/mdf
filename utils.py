import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import torch
import torch.nn.functional as F
from matplotlib.ticker import MaxNLocator

def psnr(pred, target):
    if F.mse_loss(pred,target)==0.0:
        return torch.tensor([100.0])
    else:
        return 10 * torch.log10(1 / F.mse_loss(pred, target))

# def log_report(epoch, iteration, loss, psnr):
#     print('Epoch {:d} | Iteration {:d} | Loss: {:>1.5f} | PSNR: {:>1.5f}'.format(epoch + 1, iteration+1, loss, psnr))

def print_params(params):
    print('Training parameters: ')
    print('\n'.join('  {} = {}'.format(k, str(v)) for k, v in params.items()))
    print()

def get_files(path):
    files = os.listdir(path)
    files = [os.path.join(path, x) for x in files]
    files.sort()
    return files

def load_train_ckpt(model, ckpt_dir):
    # Get latest
    files = os.listdir(ckpt_dir)
    if not files:
        return
    files = [os.path.join(ckpt_dir, x) for x in files]
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    ckpt_path = files[-1]
    # Load ckpt
    print('Loading:', ckpt_path)
    state_dict = torch.load(ckpt_path)
    state_dict.pop('params')
    model.load_state_dict(state_dict)

def load_test_ckpt(ckpt_path):
    state_dict = torch.load(ckpt_path)
    params = state_dict['params']
    state_dict.pop('params')
    return state_dict, params

def save_model_stats(model, params, ckpt_fname, stats):
    ckpt_path = os.path.join(params['ckpt_dir'], ckpt_fname)
    state_dict = model.state_dict()
    state_dict['params'] = params
    torch.save(state_dict, ckpt_path)
    # Save stats
    stats_path = os.path.join(params['stats_dir'], 'stats.json')
    with open(stats_path, 'w') as fp:
        json.dump(stats, fp, indent=2)


def plot_per_check(stats_dir, title, measurements, y_label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, len(measurements) + 1), measurements)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Checkpoint')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()

    fname = '{}.png'.format(title.replace(' ', '-').lower())
    plot_fname = os.path.join(stats_dir, fname)
    plt.savefig(plot_fname, dpi=200)
    plt.close()


class AvgMeter(object):
    """Acumulate and compute average"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def reinhard_tonemap(img):
    return img / (1 + img)
