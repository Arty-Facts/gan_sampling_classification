import torch, torchvision
import matplotlib.pyplot as plt
import numpy as np
from models import ResNet18_LowRes
import tqdm
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import pathlib
import datasets as ds
from torch.utils.data import DataLoader
import db_logger
from copy import deepcopy
import random


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
    
    acc = (np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)).item()
    
    precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average=None)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(true, pred, average="macro")
    
    
    classes = range(len(precision))
    
    fig, axes = plt.subplots(6, 1, figsize=(min(num_clasees, 20), min(6*num_clasees, 120)))
    
    plot_confusion_matrix(conf_mat=conf_matrix, show_absolute=True, show_normed=True, colorbar=True, figure=fig, axis=axes[0])
    axes[0].set_title("Confusion Matrix")
    
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
    aug_level = conf.get('aug_level', 1)

    batch_size = conf.get('batch_size', 64)
    lr = conf.get('lr', 5e-4)
    balanced_lvl = conf.get('balanced_lvl', 0)
    reduce_level = conf.get('reduce_level', 1)
    root_dir = "/mnt/data/arty/data/gan_sampling/"
    data_dir = pathlib.Path(root_dir)/"data"
    data_dir.mkdir(exist_ok=True, parents=True)
    data_blob = ds.get_dataset(dataset, f"{root_dir}/data",reduce_level=reduce_level)
    kimg = conf.get('kimg', 10_000)

    result_dir = f"{root_dir}results/{dataset}_rlvl{reduce_level}_blvl{balanced_lvl}_alvl{aug_level}"
    result_dir = pathlib.Path(result_dir)
    result_dir.mkdir(exist_ok=True, parents=True)

    train_transform, test_transform = ds.get_dataset_aug(dataset=dataset, aug_lvl=aug_level)
    train_loader = DataLoader(data_blob["train"], batch_sampler=ds.InfiniteClassSampler(data_blob["train"], batch_size=batch_size, balanced_lvl=balanced_lvl))
    test_loader =  DataLoader(data_blob["test"], batch_size=batch_size, shuffle=False)
    num_classes = data_blob['num_classes']
    logger = db_logger.DB_Logger(f"{root_dir}/baseline.db")
    exp_id = logger.register_experiment(f"{dataset}_rlvl{reduce_level}_blvl{balanced_lvl}_alvl{aug_level}", dataset=dataset, numb_classes=num_classes, seed=seed)
    run_id = logger.get_next_run_id(exp_id)
    images = []
    for l in range(num_classes):
        for _ in range(10):
            index = data_blob["train"].get_random_index_for_label(l)
            img, _ = data_blob["train"][index]
            images.append(train_transform(img)) 
            
    grid = torchvision.utils.make_grid(images, nrow=10, padding=1, normalize=True)
    train_image = torchvision.transforms.functional.to_pil_image(grid)
    

    images = []
    for l in range(num_classes):
        for _ in range(10):
            index = data_blob["test"].get_random_index_for_label(l)
            img, _ = data_blob["test"][index]
            images.append(test_transform(img)) 
            
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
        img = train_transform(img.to(device))
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
                baseline_model, test_loader, test_transform, loss_fn,
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
    
    res = evaluate_model(baseline_model, test_loader, test_transform, loss_fn, device, result_dir, num_classes, data_blob["train"].data_per_class, seen_img) 
    logger.report_result(exp_id, run_id, seen_img, res['acc'], res['f1'], res['precision'], res['recall'], res['conf_matrix'])
    try:
        torch.save(baseline_model.state_dict(), result_dir/ f"model_{seen_img}.pth")
    except Exception as e:
        print(e)
                          


def runner(func, conf, device_id):
    curr_conf = deepcopy(conf)
    curr_conf['device'] = device_id 
    return func(curr_conf)
if __name__ == '__main__':
    import device_info as di
    import ops_utils 

    jobs = []
    for aug_level in [3]:
        for reduce_level in [1, 2, 3, 0]:
            for dataset in ['BloodMNIST','PathMNIST','OrganCMNIST',]:#ds.ALL_DATASETS:
                for balanced_lvl in [0, 1]:
                    jobs.append((main, {
                        'dataset': dataset,
                        'balanced_lvl': balanced_lvl,
                        'reduce_level': reduce_level,
                        'aug_level': aug_level,
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
    random.shuffle(jobs)
    print(f'Running {len(jobs)} jobs...')
    ops_utils.parallelize(runner, jobs, gpu_nodes, verbose=True, timeout=60*60*24*14)
        