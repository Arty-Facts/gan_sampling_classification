import numpy as np
import scipy 
import torch
import numba as nb
from metrics.inception import InceptionV3

from torch.utils.data import random_split

def generate_samples(generator, samples, device):
    return [generator(torch.randn(1, generator.z_dim, device=device)).detach().cpu() for _ in range(samples)]

def get_samples(data, samples):
    if len(data) < samples:
        raise ValueError("Not enough data samples")
    if len(data) > samples:
        data , _ = random_split(data, [samples, len(data) - samples])
    return [torch.unsqueeze(data[i][0], 0) for i in range(samples)]

def get_activations(images, model, device='cpu'):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : List of image
    -- model       : Instance of inception model
    -- device      : Device to run calculations
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()
    pred_arr = []
    max_pool = torch.nn.AdaptiveMaxPool2d((1, 1))
    avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    for image in images:
        image = image.to(device)

        with torch.no_grad():
            pred = avg_pool(model.embed(image))
        pred = pred.view(-1).cpu().numpy()
        pred_arr.append(pred)

    return np.array(pred_arr)



def get_inception_activations(images, model, device='cpu'):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : List of image
    -- model       : Instance of inception model
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()
    pred_arr = []

    avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
    for image in images:
        image = image.to(device)

        with torch.no_grad():
            pred = model(image)[0]

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = avg_pool(pred)
        pred = pred.view(-1).cpu().numpy()
        pred_arr.append(pred)

    return np.array(pred_arr)


# nb.types.List(nb.types.Array(nb.types.float32, 2, 'C'))
# @nb.jit(nopython=True, cache=True)
def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    mu_diff = mu1 - mu2

    # Product might be almost singular
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)[0]
    if not np.isfinite(covmean).all():
        print('product of cov matrices is singular')
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            return np.nan
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (mu_diff.dot(mu_diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def inception_activation_statistics(images, model, device='cpu'):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : List of image 
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_inception_activations(images, model, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def activation_statistics(images, model, device='cpu',):
    """Calculation of the statistics.
    Params:
    -- images      : List of image 
    -- model       : Instance of inception model
    -- device      : Device to run calculations
    Returns:
    -- mu    : The mean over samples of the activations.
    -- sigma : The covariance matrix of the activations.
    """

    act = get_activations(images, model, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

@nb.njit(cache=True)
def numpy_cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def torch_cosine_similarity(a, b):
    similarity = torch.nn.functional.cosine_similarity(a, b, dim=0)
    return similarity

def cosine_similarity(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return numpy_cosine_similarity(a, b)
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch_cosine_similarity(a, b)
    else:
        raise TypeError(f'imgs1 and imgs2 must be both numpy.ndarray or torch.Tensor but got {type(imgs1)} and {type(imgs2)}')


@nb.njit(cache=True)
def numpy_compute_mean_and_variance(images):
    # Concatenate the images into a single NumPy array
    image_array = np.stack(images, axis=0)
    # Compute the mean and variance along the appropriate axes
    mean = np.mean(image_array, axis=(0))
    variance = np.var(image_array, axis=(0))

    return mean, variance

def torch_compute_mean_and_variance(images):
    # Concatenate the images into a single NumPy array
    image_array = torch.stack(images, dim=0)
    # Compute the mean and variance along the appropriate axes
    mean = torch.mean(image_array, dim=0)
    variance = torch.var(image_array, dim=0)

    return mean, variance

def compute_mean_and_variance(images):
    if isinstance(images, np.ndarray):
        return numpy_compute_mean_and_variance(images)
    elif isinstance(images, torch.Tensor):
        return torch_compute_mean_and_variance(images)
    else:
        raise TypeError(f'imgs1 and imgs2 must be both numpy.ndarray or torch.Tensor but got {type(imgs1)} and {type(imgs2)}')


@nb.njit(cache=True)
def numpy_mean_var_cosine_similarity(imgs1, imgs2):
    mean1, var1 = numpy_compute_mean_and_variance(imgs1)
    mean2, var2 = numpy_compute_mean_and_variance(imgs2)

    combine1 = np.concatenate((mean1, var1))
    combine2 = np.concatenate((mean2, var2))
    coefficient_of_variation1 = np.divide(var1, mean1)
    coefficient_of_variation2 = np.divide(var2, mean2)
    variance_to_mean_ratio1 = np.divide(var1*var1, mean1)
    variance_to_mean_ratio2 = np.divide(var2*var2, mean2)

    mean_cos = cosine_similarity(mean1.reshape(-1), mean2.reshape(-1))
    var_cos = cosine_similarity(var1.reshape(-1), var2.reshape(-1))
    stack_cos = cosine_similarity(combine1.reshape(-1), combine2.reshape(-1))
    cov_cos = cosine_similarity(coefficient_of_variation1.reshape(-1), coefficient_of_variation2.reshape(-1))
    vmr_cos = cosine_similarity(variance_to_mean_ratio1.reshape(-1), variance_to_mean_ratio2.reshape(-1))

    return mean_cos, var_cos, stack_cos, cov_cos, vmr_cos

def torch_mean_var_cosine_similarity(imgs1, imgs2):
    mean1, var1 = torch_compute_mean_and_variance(imgs1)
    mean2, var2 = torch_compute_mean_and_variance(imgs2)
    combine1 = torch.cat((mean1, var1))
    combine2 = torch.cat((mean2, var2))
    coefficient_of_variation1 = torch.div(var1, mean1)
    coefficient_of_variation2 = torch.div(var2, mean2)
    variance_to_mean_ratio1 = torch.div(var1*var1, mean1)
    variance_to_mean_ratio2 = torch.div(var2*var2, mean2)

    mean_cos = cosine_similarity(mean1.view(-1), mean2.view(-1))
    var_cos = cosine_similarity(var1.view(-1), var2.view(-1))
    stack_cos = cosine_similarity(combine1.view(-1), combine2.view(-1))
    cov_cos = cosine_similarity(coefficient_of_variation1.view(-1), coefficient_of_variation2.view(-1))
    vmr_cos = cosine_similarity(variance_to_mean_ratio1.view(-1), variance_to_mean_ratio2.view(-1))

    return mean_cos, var_cos, stack_cos, cov_cos, vmr_cos

def mean_var_cosine_similarity(imgs1, imgs2):
    if isinstance(imgs1[0], np.ndarray) and isinstance(imgs2[0], np.ndarray):
        return numpy_mean_var_cosine_similarity(imgs1, imgs2)
    elif isinstance(imgs1[0], torch.Tensor) and isinstance(imgs2[0], torch.Tensor):
        return torch_mean_var_cosine_similarity(imgs1, imgs2)
    else:
        raise TypeError(f'imgs1 and imgs2 must be both numpy.ndarray or torch.Tensor but got {type(imgs1[0])} and {type(imgs2[0])}')
    

def frechet_inception_distance(dataset1, dataset2, dims=2048, device='cpu'):
    """Computes the Frechet Inception Distance (FID) to evalulate the quality of
    generated images. The FID metric calculates the distance between two
    distributions of images."""

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    d1_images = [torch.unsqueeze(dataset1[i], 0) for i in range(len(dataset1))]
    d2_images = [torch.unsqueeze(dataset2[i], 0) for i in range(len(dataset2))]

    mean_d1, std_d1 = inception_activation_statistics(d1_images, model, device)
    mean_d2, std_d2 = inception_activation_statistics(d2_images, model, device)

    fd = frechet_distance(mean_d1, std_d1, mean_d2, std_d2)
    return fd