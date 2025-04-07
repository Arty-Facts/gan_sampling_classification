import torch, torchvision
import stylegan.networks_stylegan3 as stylegan3
import stylegan.networks_stylegan2 as stylegan2
import stylegan.training_loop_gan as training_loop_gan
import datasets as ds
import pathlib



def main(config=None):
    if config is None:
        config = {}
    data = ds.get_dataset(config.get("dataset", "CIFAR10"), "data")
    
    img_resolution = data["im_size"]
    channel = data["channel"]
    c_dim = data["num_classes"]
    z_dim = config.get("z_dim", 512)
    w_dim = config.get("w_dim", 512)
    rank = config.get("rank", 0) # what gpu to use
    outdir = config.get("outdir", "results_stylegan3")
    total_kimg = config.get("total_kimg", 25000)
    batch_size = config.get("batch_size", 16)
    resume_pkl = config.get("resume_pkl", None)
    fourier_features = config.get("fourier_features", None)
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    
    train_data = data["train"]
    G = stylegan3.Generator(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, img_channels=channel, img_resolution=img_resolution, fourier_features=fourier_features)
    D = stylegan2.Discriminator(c_dim, img_channels=channel, img_resolution=img_resolution, fourier_features=fourier_features)
    G_ema = training_loop_gan.train(G, D, train_data, rank=rank, run_dir=outdir, resume_pkl=resume_pkl, total_kimg=total_kimg, batch_size=batch_size, batch_gpu=batch_size)

    torch.save(G.state_dict(), f"{outdir}/G.pth")
    torch.save(D.state_dict(), f"{outdir}/D.pth")
    torch.save(G_ema.state_dict(), f"{outdir}/G_ema.pth")


if __name__ == "__main__":
    main() 
