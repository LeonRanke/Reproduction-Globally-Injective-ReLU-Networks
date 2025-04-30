import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm


import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms




def test_minimal_expansivity(c: float, num_steps: int=50):
    img_size = 28   # Siz of images
    num_viz = 5
    selected_idicies = [[0, 2], [2, 2], [4, 2]] # for reconstruction

    # Create a random tensor using Gaussian noise with the specified size
    n = img_size**2
    m = int(c * n)
    W = torch.randn((m, n))

    # Get the Mnist dataset
    root = os.path.join(os.getcwd(), "data")
    transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(root, train=False, download=True, transform=transform)

    # Get: x0, x1, x2 and x1+x2
    x0 = -torch.mean(W / W.norm(dim=1, keepdim=True), dim=0).reshape(1, img_size, img_size)
    x1, _ = mnist_dataset[5]
    x2, _ = mnist_dataset[1]
    x12 = x1*0.5 + x2*0.5

    # Linearly interpolate between x0 and x1 and x0 and x1+x2
    alphas = torch.linspace(0, 1, num_steps).view(-1, 1, 1, 1)  # shape (num_steps, 1, 1, 1)
    inter_x1 = (1 - alphas) * x0 + alphas * x1   # Interpolate between x0 and x1
    inter_x2 = (1 - alphas) * x2 + alphas * x12  # Interpolate between x2 and x1+x2

    # Interpolta between the two interpolations to crate grid of images
    inter_images = torch.zeros((num_steps, num_steps, *x1.shape))
    for i in range(num_steps):
        inter_images[i] = (1 - alphas) * inter_x1[i] + alphas * inter_x2[i]


    # Compute the Result of a Relu for each image
    nums_pos = torch.zeros(num_steps, num_steps, 1,  dtype=torch.int32)
    sucsess = torch.zeros(num_steps, num_steps, 1,  dtype=torch.bool)
    for i in range(num_steps):
        for j in range(num_steps):
            image = inter_images[i, j, :, :]
            x = image.flatten().unsqueeze(0)
        
            # forward Relu layer
            y = F.relu(torch.matmul(x, W.T))
            
            # Test injectivity of sample
            num_pos = (y > 0).sum()
            nums_pos[i, j] = num_pos
            sucsess[i, j] = num_pos > n


    # Plotting the interpolated images
    _, axs = plt.subplots(num_viz, num_viz, figsize=(15, 15))
    for i in range(num_viz):
        for j in range(num_viz):
            cmap = "Greens_r" if sucsess[i, j] else "Reds_r"
            axs[j, i].imshow(inter_images[i, j, :, :].permute(1, 2, 0), cmap=cmap, vmin=0, vmax=1)
            axs[j, i].axis("off")
            axs[j, i].set_title(nums_pos[i, j].item())

            # Add a white border for the selected indices
            if [i, j] in selected_idicies:
                rect = patches.Rectangle((0, 0), img_size-1, img_size-1, linewidth=4, edgecolor='yellow', facecolor='none')
                axs[j, i].add_patch(rect)
    save_name = f"Example_MNIST.png"
    save_path = os.path.join(os.getcwd(), "img", "Figure_6", save_name)
    plt.savefig(save_path)

    # -------------------------------------------------------------------------
    # Reconstructing selected images
    num_tries = 4
    noise_db = 20 # dB
    
    
    # Invert the weights W
    W_inv = torch.linalg.pinv(W)

    _, axs = plt.subplots(len(selected_idicies), num_tries + 1, figsize=(15, 5)) 
    for i, index in enumerate(selected_idicies):
        # Get the Original image
        x = inter_images[index[0], index[1]]
        axs[i, 0].imshow(x.permute(1, 2, 0))
        axs[i, 0].set_title(nums_pos[index[0], index[1]].item())
        axs[i, 0].axis("off")

        # Pass image throgh ReLU Layer
        x = x.flatten().unsqueeze(0)
        y = F.relu(torch.matmul(x, W.T))

        # Attempt reconstruction
        for j in range(num_tries):
           # Gerate 10db noise and add it to x
           signal_power = torch.mean(y ** 2)
           noise_std = torch.sqrt(signal_power / (10 ** (noise_db / 10)))
           noise = noise_std * torch.randn_like(y)
           noisy_y = y + noise

           # reconstruuct x from a noisy y
           recon_x = torch.matmul(noisy_y, W_inv.T)
           recon_x = recon_x.reshape(1, img_size, img_size)
           axs[i, j+1].imshow(recon_x.permute(1, 2, 0))
           axs[i, j+1].axis("off")
    save_name = f"Example_MNIST_Reconstructions.png"
    save_path = os.path.join(os.getcwd(), "img", "Figure_6", save_name)
    plt.savefig(save_path)


def prob_injectivity(mode):
    ns = [500, 1000, 5000]
    if mode == "Paper":
        cs = torch.linspace(2, 4.5, 50)
    else:
        cs = torch.linspace(1, 4.5, 50)
    num_tries = 100
    
    times_injective = torch.zeros((len(ns), *cs.shape), dtype=torch.int32)
    pos_components = torch.zeros((len(ns), *cs.shape, num_tries), dtype=torch.float64)

    for j, n in enumerate(ns):
        print(f"n = {n}")
        for i, c in tqdm(enumerate(cs), total=len(cs)):
            for k in range(num_tries):
                # Create a random tensor using Gaussian noise with the specified size
                m = int(c * n)
                W = torch.randn((m, n))

                # Get x0
                if mode == "Paper":
                    x = -torch.mean(W / W.norm(dim=1, keepdim=True), dim=0).unsqueeze(0)
                else:
                    x = torch.rand(n).unsqueeze(0)
            
                # Callculate y = Relu(Wx)
                y = F.relu(torch.matmul(x, W.T))

                # dtermine number of positive components
                num_pos = (y > 0).sum()
                pos_components[j, i, k] = num_pos

                if num_pos >= n:
                    times_injective[j][i] += 1


    # Plot number of positive components
    pos_components = torch.mean(pos_components, dim=-1)
    plt.figure(figsize=(12, 7))
    plt.yscale('log')
    for i in range(len(ns)):
        plt.plot(cs, pos_components[i], "-", label=f"$n = {ns[i]}$")
        plt.scatter(cs, pos_components[i])
        plt.axhline(y=ns[i], color='black', linestyle=':',)
        plt.text(x=9, y=ns[i], s=str(ns[i]), ha="left", va="center")
    
    plt.grid()
    plt.title("Number of positive components in $ReLU(Wx)$ with $W \in \mathbb{R}^{cn x n}$")
    plt.legend(loc="upper left")
    plt.yticks([100, 500, 1000, 5000])
    plt.xlabel("Expansivity factor, $c$")
    plt.ylabel("$|\{i|<\omega_i, x> > 0\}|$")

    end = "" if mode == "Paper" else "_Random_x"
    save_name = f"Number_pos_Components{end}.png"
    save_path = os.path.join(os.getcwd(), "img", "Figure_7", save_name)
    plt.savefig(save_path)

    
    # Plot probability of layer beeing injective
    times_injective = times_injective / num_tries
    plt.figure(figsize=(7, 7))
    for i in range(len(ns)):
        plt.plot(cs, times_injective[i], label=f"$n = {ns[i]}$")
    
    plt.grid()
    plt.title("Probability of $ReLU(Wx)$ with $W \in \mathbb{R}^{cn x n}$ beeing injective")
    plt.legend(loc="upper left")
    plt.xlabel("Expansivity factor, $c$")
    plt.ylabel("Prob. of $x$ havig a DSS")
    
    end = "" if mode == "Paper" else "_Random_x"
    save_name = f"Probability_of_Injectivity{end}.png"
    save_path = os.path.join(os.getcwd(), "img", "Figure_7", save_name)
    plt.savefig(save_path)





def main():
    parser = argparse.ArgumentParser(description="Visualize the weges created in a single ReLU layer")
    parser.add_argument("--c", default=2.1, type=float, required=False, help="Expansivity of the ReLu layer defaoult=2.1 (from paper)")
    args = parser.parse_args()

    test_minimal_expansivity(c=args.c)
    prob_injectivity(mode="Paper")
    prob_injectivity(mode="Random")

if __name__ == "__main__":
    main()
