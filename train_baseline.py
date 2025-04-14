import torch, torchvision
import torchvision.transforms.v2 as transforms_v2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from metrics.utils import frechet_inception_distance
from models import ResNet18_LowRes
import tqdm
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.decomposition import PCA
import ood_detectors.vision as vision_ood
import ood_detectors.likelihood as likelihood
import ood_detectors.residual as residual
import ood_detectors.eval_utils as eval_utils
import pathlib
import datasets as ds
from torch.utils.data import DataLoader
import db_logger
from copy import deepcopy
import random
    
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
    
def plot(scores, tile, out_dir='figs', verbose=True, names=None, extra=None):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    if verbose:
        print('Generating plots...')


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
        sns.kdeplot(data=score, bw_adjust=.2, ax=ax, label=f'{index}: auc: {eval_utils.auc(score, scores[0]):.2f}, mean: {np.mean(score):.2} s: {len(score)} ')
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
    plt.savefig(out_dir/filename, bbox_inches='tight')

@torch.no_grad()
def evaluate_model(model, valid_loader, transform, loss_fn, device, out_dir, num_clasees, train_lables, seen_img):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    conf_matrix = torch.zeros(num_clasees, num_clasees)
    model.eval()
    valid_loss = 0
    pred = []
    true = []
    probas = []
    
    for img, labels in valid_loader:
        img = transform(img.to(device))
        labels = labels.to(device)
        out = model(img)
        loss = loss_fn(out, labels)
        valid_loss += loss.item()
        
        for i, j in zip(labels.cpu(), out.argmax(dim=1).cpu()):
            conf_matrix[i, j] += 1
        pred.append(out.argmax(dim=1).cpu().numpy())
        true.append(labels.cpu().numpy())
        probas.append(torch.softmax(out, dim=1).cpu().numpy())
    
    valid_loss /= len(valid_loader)
    conf_matrix = conf_matrix.cpu().numpy()
    pred = np.concatenate(pred)
    true = np.concatenate(true)
    probas = np.concatenate(probas)
    
    # Accuracy
    acc = (np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)).item()
    
    # Precision, Recall, F1-score (Per Class)
    precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average=None)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(true, pred, average="macro")
    
    
    classes = range(len(precision))
    
    print(f'{out_dir} valid loss: {valid_loss}, acc: {acc}, f1:{macro_f1}, precision:{macro_precision}, recall:{macro_recall}')
    
    # Plot Confusion Matrix and Per-Class Metrics in a 6x1 Subplot
    fig, axes = plt.subplots(6, 1, figsize=(min(num_clasees, 20), min(6*num_clasees, 120)))
    # fig.suptitle("Confusion Matrix and Per-Class Evaluation Metrics")
    
    # Confusion Matrix
    plot_confusion_matrix(conf_mat=conf_matrix, show_absolute=True, show_normed=True, colorbar=True, figure=fig, axis=axes[0])
    axes[0].set_title("Confusion Matrix")
    
    # Per-Class Metrics
    metrics = [
        ("Accuracy", np.diag(conf_matrix) / conf_matrix.sum(axis=1), acc), 
        ("F1-score", f1, macro_f1), 
        ("Precision", precision, macro_precision), 
        ("Recall", recall, macro_recall),
        ]
    sns.barplot(x=classes, y=train_lables, ax=axes[1])
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Train Samples")
    axes[1].set_title(f"Train Samples per Class")
    axes[1].grid()
    fig.tight_layout()
    
    for ax, (metric,values, macro) in zip(axes[2:], metrics):
        sns.barplot(x=classes, y=values, ax=ax)
        ax.set_xlabel("Class")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} per Class, Macro: {macro:.3f}")
        ax.grid()
        # ax.set_ylim(max(values.min()-0.01, 0), min(values.max()+0.01, 1))
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', fontsize=10, color='black')
    try:
        fig.savefig(out_dir/f"metrics_{seen_img}.png")
    except Exception as e:
        print(e)
    plt.close()
    
    return {
        "valid_loss": valid_loss,
        "acc": acc,
        "f1": macro_f1,
        "precision": macro_precision,
        "recall": macro_recall,
        "conf_matrix": conf_matrix,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_score_per_class": f1.tolist(),
    }
        

def main(conf):
    dataset = conf.get('dataset', 'BloodMNIST')
    device = conf.get('device', 'cuda')
    seed = conf.get('seed', None)
    aug = conf.get('aug', True)

    batch_size = conf.get('batch_size', 64)
    lr = conf.get('lr', 5e-4)
    balanced = conf.get('balanced', True)
    reduce_level = conf.get('reduce_level', 1)
    root_dir = "/mnt/data/arty/data/gan_sampling/"
    data_dir = pathlib.Path(root_dir)/"data"
    data_dir.mkdir(exist_ok=True, parents=True)
    data_blob = ds.get_dataset(dataset, f"{root_dir}/data",reduce_level=reduce_level)
    kimg = conf.get('kimg', 10_000)
    # if reduce_level == 2:
    #     kimg *= 10
    # if reduce_level == 3:
    #     kimg *= 2
    result_dir = f"{root_dir}results/{dataset}_rlvl{reduce_level}_b{balanced}_aug{aug}_{kimg}"
    result_dir = pathlib.Path(result_dir)
    result_dir.mkdir(exist_ok=True, parents=True)
    # encoder_name = conf.get('encoder', 'dinov2')
    # encoder = vision_ood.get_encoder(encoder_name)
    # embedding_size = 768
    # encoder.eval()
    # encoder.to(device)
    # encoder_transform = get_encoder_transform(encoder, device=device)

    # preprocessing
    data_transform = transforms.Compose([
        transforms_v2.RandomRotation(16),
        transforms_v2.RandomHorizontalFlip(),
        # transforms_v2.RandomVerticalFlip(),
        transforms_v2.RGB(),
        # transforms_v2.RandomAutocontrast(),
        # transforms_v2.ColorJitter(),
        # transforms_v2.GaussianNoise(),
        transforms_v2.Normalize(mean=[.5], std=[.5]),
    ])
    val_data_transform = transforms.Compose([
        transforms_v2.RGB(),
        transforms_v2.Normalize(mean=[.5], std=[.5])
    ])

    if not aug:
        data_transform = val_data_transform
    train_loader = DataLoader(data_blob["train"], batch_sampler=ds.InfiniteClassSampler(data_blob["train"], batch_size=batch_size, balanced=balanced))
    test_loader =  DataLoader(data_blob["test"], batch_size=batch_size, shuffle=False)
    num_classes = data_blob['num_classes']
    logger = db_logger.DB_Logger(f"{root_dir}/baseline.db")
    exp_id = logger.register_experiment(f"{dataset}_rlvl{reduce_level}_b{balanced}_aug_{aug}", dataset=dataset, numb_classes=num_classes, seed=seed)
    run_id = logger.get_next_run_id(exp_id)
    images = []
    for l in range(num_classes):
        for _ in range(10):
            index = data_blob["train"].get_random_index_for_label(l)
            img, _ = data_blob["train"][index]
            images.append(data_transform(img)) 
            
    grid = torchvision.utils.make_grid(images, nrow=10, padding=1, normalize=True)
    train_image = torchvision.transforms.functional.to_pil_image(grid)
    

    images = []
    for l in range(num_classes):
        for _ in range(10):
            index = data_blob["test"].get_random_index_for_label(l)
            img, _ = data_blob["test"][index]
            images.append(val_data_transform(img)) 
            
    grid = torchvision.utils.make_grid(images, nrow=10, padding=1, normalize=True)
    test_image = torchvision.transforms.functional.to_pil_image(grid)
    try:
        test_image.save(result_dir/"test_images.png")
        train_image.save(result_dir/"train_images.png")
    except Exception as e:
        print(e)

    loss_fn = torch.nn.CrossEntropyLoss()
    baseline_model = ResNet18_LowRes(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=lr)
    # lrs = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=((1000*kimg)//batch_size)+1)
    baseline_model.train()
    seen_img=0
    eval_interval = (np.concat([np.linspace(10000, 100_000, 10), np.linspace(110_000, 1_000_000, 40), np.linspace(1_100_000, 10_000_000, 40), np.linspace(11_000_000, 100_000_000, 100)])).astype(np.int32).tolist()
    next_eval = eval_interval.pop(0)
    results = []
    for img, l in train_loader:
        # print(img.shape, l.shape)
        img = data_transform(img.to(device))
        l = l.to(device)
        optimizer.zero_grad()
        out = baseline_model(img)
        # print(out.shape)
        loss = loss_fn(out, l)
        loss.backward()
        optimizer.step()
        seen_img += img.shape[0]
        if seen_img >= next_eval:
            res = evaluate_model(
                baseline_model, test_loader, val_data_transform, loss_fn,
                device, result_dir, num_classes, data_blob["train"].data_per_class, seen_img
            )
            logger.report_result(exp_id, run_id, seen_img, res['acc'], res['f1'], res['precision'], res['recall'], res['conf_matrix'])
            try:
                torch.save(baseline_model.state_dict(), result_dir/ f"model_{seen_img}.pth")
            except Exception as e:
                print(e)
            if len(results) > 50 and min(results[-20:]) > res['f1'] and max(results) > max(results[-20:]): #no progress:
                break

            results.append(res['f1'])
            next_eval = eval_interval.pop(0)
        if seen_img >= kimg*1000:
            break
        # lrs.step()
    
    res = evaluate_model(baseline_model, test_loader, val_data_transform, loss_fn, device, result_dir, num_classes, data_blob["train"].data_per_class, seen_img) 
    logger.report_result(exp_id, run_id, seen_img, res['acc'], res['f1'], res['precision'], res['recall'], res['conf_matrix'])
    try:
        torch.save(baseline_model.state_dict(), result_dir/ f"model_{seen_img}.pth")
    except Exception as e:
        print(e)
                          

    # print("Results:")
    # for name, score in result.items():
    #     print(f"{name}: {score}")
    # embs = {
    #     "train": [[] for _ in range(num_classes)],
    #     "test": [[] for _ in range(num_classes)],
    # }
            
    # data = {
    #     "train": [[] for _ in range(num_classes)],
    #     "test": [[] for _ in range(num_classes)],
    # }
    # scores = {
    #     "train": [[None]*num_classes for _ in range(num_classes)],
    #     "test": [[None]*num_classes for _ in range(num_classes)],
    # }
    # for img, l in data_blob["train"]:
    #     data["train"][l].append(img)

    # for img, l in data_blob["test"]:
    #     data["test"][l].append(img)

    # print("Embed imgs")
    # for name in ["train", "test"]:
    #     for c, d in enumerate(data[name]):
    #         loader = torch.utils.data.DataLoader(d, batch_size=16, shuffle=False)
    #         with torch.no_grad():
    #             for img in loader:
    #                 img = encoder_transform(img.to(device))
    #                 emb = encoder(img)
    #                 embs[name][c].append(emb)
    #         embs[name][c] = torch.concatenate(embs[name][c])


    # print("fitt ood")
    # ood_detectors = [residual.ResidualX(dims=(0.3, 0.5), k=3, subsample=0.8) for _ in range(num_classes)]

    # ood_detector_name = ood_detectors[0].name
    # for i, ood_detector in enumerate(ood_detectors):
    #     ood_detector.to(device)
    #     ood_detector.fit(embs["train"][i], batch_size=1000, n_epochs=10, verbose=False)
    #     ood_detector.to("cpu")

    # print("calc scores")
    # for i, ood_detector in enumerate(ood_detectors):
    #     for j in range(8):
    #         ood_detector.to(device)
    #         for name in ["train", "test"]:
    #             score = ood_detector.predict(embs[name][j], batch_size=1000, verbose=False)
    #             score = torch.from_numpy(score)
    #             scores[name][i][j] = score
    #         ood_detector.to("cpu")
    # for i in range(num_classes):
    #     scores_class = [scores[name][i][i].numpy() for name in scores.keys()] + [scores[name][i][j].numpy() for j in range(num_classes) if i != j]
    #     names = [n for n in scores.keys()] + [f"test{j}" for j in range(num_classes) if i != j]
    #     plot(scores_class,f"score_plot_{ood_detector_name}_{encoder_name}_{i}", out_dir=result_dir,names=names)
    


def runner(func, conf, device_id):
    curr_conf = deepcopy(conf)
    curr_conf['device'] = device_id 
    return func(curr_conf)
if __name__ == '__main__':
    import device_info as di
    import ops_utils 

    jobs = []
    for aug in [True, False]:
        for reduce_level in [1, 2, 3, 0]:
            for dataset in ['OrganCMNIST',]:# ['BloodMNIST','PathMNIST','OrganCMNIST',]:#ds.ALL_DATASETS:
                for balanced in [True, False]:
                    jobs.append((main, {
                        'dataset': dataset,
                        'balanced': balanced,
                        'reduce_level': reduce_level,
                        'aug': aug,
                        }))

    device_info = di.Device()


    gpu_nodes = []
    mem_req = 1.2
    max_per_gpu = 4
    if len(device_info) == 1: # we are on wood
        max_per_gpu = 32
    for id, gpu in enumerate(device_info):
        if gpu.mem.free > mem_req:
            use_gpu = int(gpu.mem.free/mem_req)
            if use_gpu > max_per_gpu:
                use_gpu = max_per_gpu
            gpu_nodes.extend([id]*use_gpu)
    if len(gpu_nodes) == 0:
        raise ValueError('No available GPU nodes')
    jobs = jobs*3
    # random.shuffle(jobs)
    print(f'Running {len(jobs)} jobs...')
    ops_utils.parallelize(runner, jobs, gpu_nodes, verbose=True, timeout=60*60*24*14)
        