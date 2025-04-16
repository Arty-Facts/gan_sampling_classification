import torch, torchvision
import torchvision.transforms.v2 as transforms_v2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from metrics.utils import frechet_inception_distance
from models import ResNet18_LowRes
import tqdm
import seaborn as sns
from sklearn.decomposition import PCA
import ood_detectors.vision as vision_ood
import ood_detectors.likelihood as likelihood
import ood_detectors.residual as residual
import ood_detectors.eval_utils as eval_utils
import pathlib
import datasets as ds
import db_logger
from copy import deepcopy
import random, json
    
class Composite(torch.nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)

    def forward(self, img):
        for t in self.transforms:
            img = t(img)
        return img
    
    def __repr__(self):
        return f'Composite({self.transforms})'
    
def get_interpolation_mode(transform):
    if isinstance(transform, transforms.Resize):
        mode = ""
        if transform.interpolation == transforms.InterpolationMode.BILINEAR:
            mode = 'bilinear'
        elif transform.interpolation == transforms.InterpolationMode.NEAREST:
            mode = 'nearest'
        elif transform.interpolation == transforms.InterpolationMode.BICUBIC:
            mode = 'bicubic'
        else:
            raise ValueError(f'Invalid interpolation mode: {transform.interpolation}')
        return mode
    else:
        raise ValueError(f'Invalid transform: {transform}')
    
def get_encoder_transform(model, device='cpu'):
    layers = [transforms_v2.RGB()]
    for t in model.transform.transforms:
        if isinstance(t, transforms.Resize):
            mode = get_interpolation_mode(t)
            layers.append(torch.nn.Upsample(size=t.size, mode=mode))
        elif isinstance(t, transforms.CenterCrop):
            layers.append(t)
        elif isinstance(t, transforms.Normalize):
            layers.append(t)
        else:
            print(f'Warning: Ignoring transform {t}')
    return Composite(layers).to(device)

def save_ood(res, dataset, reduce_level, balanced_lvl, aug_level, encoder_name, size, prefix):
    filename = f"ood_metric.jsonl"

    data = {
        "score": res,
        "dataset": dataset,
        "reduce_level": reduce_level,
        "balanced_level": balanced_lvl,
        "aug_level": aug_level,
        "encoder": encoder_name,
        "size": size,
        "type": prefix,
    }

    with open(filename, 'a') as f:
        json.dump(data, f)
        f.write('\n')

    print(f"Saved {len(res)} entries with metadata to {filename}")

def load_ood(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data

def plot(scores, tile, out_dir='figs', verbose=True, names=None, extra=None):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    res = []


    # Create a figure with subplots
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))  # Adjust the size as needed
    fig.suptitle(f'{tile} Evaluation')

    def add_shadow(ax, data):
        if data.var() > 1e-6:
            l = ax.lines[-1]
            x = l.get_xydata()[:,0]
            y = l.get_xydata()[:,1]
            ax.fill_between(x,y, alpha=0.1)
            # Calculate and plot the mean
            mean_value = np.mean(data)
            line_color = l.get_color()
            ax.axvline(mean_value, color=line_color, linestyle=':', linewidth=1.5)

    if names is None:
        it = enumerate(scores)
    else:
        it = zip(names, scores)
    for index, score in it:
        auc = eval_utils.auc(score, scores[0])
        mean = np.mean(score).item()
        res.append((index, auc, mean))
        sns.kdeplot(data=score, bw_adjust=.2, ax=ax, label=f'{index}: auc: {auc:.2f}, mean: {mean:.2} s: {len(score)} ')
        add_shadow(ax, score)

    if extra is not None:
        for name, value in extra:
            ax.axvline(value, color='black', linestyle=':', linewidth=2.5)
            ax.text(value, 0, name, rotation=90, verticalalignment='bottom', horizontalalignment='center')

    ax.set_title('Density Plots')
    ax.set_xlabel('bits/dim')
    ax.set_ylabel('Density')

    ax.legend()


    # Save the figure
    filename = f"{tile}.png"
    if verbose:
        print('Generating plots...', out_dir/filename)

    plt.savefig(out_dir/filename, bbox_inches='tight')
    return res

def main(conf):
    dataset = conf.get('dataset', 'BloodMNIST')
    device = conf.get('device', 'cuda')
    seed = conf.get('seed', None)
    aug_level = conf.get('aug_level', 1)
    super_sample = conf.get('super_sample', 10)
    if aug_level == 0:
        super_sample = 1
    balanced_lvl = conf.get('balanced_lvl', 0)
    reduce_level = conf.get('reduce_level', 1)
    # dims = conf.get('dims', 0.5)
    size = conf.get('size', 224)
    root_dir = "/mnt/data/arty/data/gan_sampling/"
    data_dir = pathlib.Path(root_dir)/"data"
    data_dir.mkdir(exist_ok=True, parents=True)
    data_blob = ds.get_dataset(dataset, f"{root_dir}/data",reduce_level=reduce_level, size=size)
    result_dir = f"{root_dir}results/{dataset}_rlvl{reduce_level}_blvl{balanced_lvl}_alvl{aug_level}"
    result_dir = pathlib.Path(result_dir)
    result_dir.mkdir(exist_ok=True, parents=True)
    encoder_name = conf.get('encoder', 'dinov2')
    encoder = vision_ood.get_encoder(encoder_name)
    embedding_size = 768
    encoder.eval()
    encoder.to(device)
    encoder_transform = get_encoder_transform(encoder, device=device)

    train_transform, test_transform = ds.get_dataset_aug(dataset=dataset, aug_lvl=aug_level)
    train_transform.transforms.pop()
    num_classes = data_blob['num_classes']

    embs = {
        "train": [[] for _ in range(num_classes)],
        "test": [[] for _ in range(num_classes)],
    }
            
    data = {
        "train": [[] for _ in range(num_classes)],
        "test": [[] for _ in range(num_classes)],
    }
    scores = {
        "train": [[None]*num_classes for _ in range(num_classes)],
        "test": [[None]*num_classes for _ in range(num_classes)],
    }
    for img, l in data_blob["train"]:
        data["train"][l].append(img)

    for img, l in data_blob["test"]:
        data["test"][l].append(img)

    print("Embed imgs")
    for c, d in enumerate(data["train"]):
        loader = torch.utils.data.DataLoader(d, batch_size=16, shuffle=False)
        with torch.no_grad():
            for _ in range(super_sample):
                for img in loader:
                    img = encoder_transform(train_transform(img.to(device)))
                    emb = encoder(img)
                    embs["train"][c].append(emb.cpu())
        embs["train"][c] = torch.concatenate(embs["train"][c])
    for c, d in enumerate(data["test"]):
        loader = torch.utils.data.DataLoader(d, batch_size=16, shuffle=False)
        with torch.no_grad():
            for img in loader:
                img = encoder_transform(img.to(device))
                emb = encoder(img)
                embs["test"][c].append(emb.cpu())
        embs["test"][c] = torch.concatenate(embs["test"][c])


    print("fitt ood")
    ood_detectors = [residual.ResidualAuto() for _ in range(num_classes)]

    for i, ood_detector in enumerate(ood_detectors):
        ood_detector.to(device)
        ood_detector.fit(embs["train"][i], embs["test"][i], torch.concatenate([embs["train"][j] for j in range(num_classes) if i != j]), batch_size=1000, n_epochs=10, verbose=False)
        ood_detector.to("cpu")

    print("calc scores")
    for i, ood_detector in enumerate(ood_detectors):
        for j in range(num_classes):
            ood_detector.to(device)
            for name in ["train", "test"]:
                score = ood_detector.predict(embs[name][j], batch_size=1000, verbose=False)
                score = torch.from_numpy(score)
                scores[name][i][j] = score
            ood_detector.to("cpu")
    for i in range(num_classes):
        scores_class = [scores[name][i][i].numpy() for name in scores.keys()] + [scores["test"][i][j].numpy() for j in range(num_classes) if i != j]
        names = [f"{n}{i}" for n in scores.keys()] + [f"test{j}" for j in range(num_classes) if i != j]
        res = plot(scores_class,f"send_plot_{ood_detectors[i].name}_{encoder_name}_{i}_{size}_test", out_dir=result_dir,names=names)
        save_ood(res, dataset, reduce_level, balanced_lvl, aug_level, encoder_name, size, "test")
        scores_class = [scores[name][i][i].numpy() for name in scores.keys()] + [scores["train"][i][j].numpy() for j in range(num_classes) if i != j]
        names = [f"{n}{i}" for n in scores.keys()] + [f"train{j}" for j in range(num_classes) if i != j]
        res = plot(scores_class,f"score_plot_{ood_detectors[i].name}_{encoder_name}_{i}_{size}_train", out_dir=result_dir,names=names)
        save_ood(res, dataset, reduce_level, balanced_lvl, aug_level, encoder_name, size, "train")
    

def runner(func, conf, device_id):
    curr_conf = deepcopy(conf)
    curr_conf['device'] = device_id 
    return func(curr_conf)
if __name__ == '__main__':
    # main({'dataset': 'BloodMNIST','size': 224,})
    # main({'dataset': 'OrganCMNIST','size': 224,})
    # main({'dataset': 'PathMNIST','size': 224,})
    import device_info as di
    import ops_utils 

    jobs = []
    for aug_level in [0, 1, 2]:
        for reduce_level in [1, 2]:
            for dataset in ['BloodMNIST','PathMNIST','OrganCMNIST',]:#ds.ALL_DATASETS:
                for encoder in ["dinov2", "clip", "vit", "resnet50"]:
                    for size in [28, 224]:
                        jobs.append((main, {
                            'dataset': dataset,
                            'encoder': encoder,
                            'reduce_level': reduce_level,
                            'aug_level': aug_level,
                            'size': size,
                            }))

    device_info = di.Device()


    gpu_nodes = []
    mem_req = 1.5
    max_per_gpu = 4
    if len(device_info) == 1: # we are on wood
        max_per_gpu = 8
    for id, gpu in enumerate(device_info):
        if gpu.mem.free > mem_req:
            use_gpu = int(gpu.mem.free/mem_req)
            if use_gpu > max_per_gpu:
                use_gpu = max_per_gpu
            gpu_nodes.extend([id]*use_gpu)
    if len(gpu_nodes) == 0:
        raise ValueError('No available GPU nodes')
    jobs = jobs
    # random.shuffle(jobs)
    print(f'Running {len(jobs)} jobs...')
    ops_utils.parallelize(runner, jobs, gpu_nodes, verbose=True, timeout=60*60*24*14)
        