import os
import argparse
import numpy as np
import matplotlib.pyplot as plt



def visualize_wedges(W: np.ndarray, save_name: str):
    m, n = W.shape

    # Test if W is full rank
    rank = np.linalg.matrix_rank(W)
    is_full_rank = rank == n
    print(f"W has a rank of: {rank}, is full rank: {is_full_rank}")

    # Generate a grid of points in R^2
    x_min, x_max, y_min, y_max = -4, 4, -4, 4
    points = np.linspace(x_min, x_max, 500)
    X, Y = np.meshgrid(points, points)
    grid = np.column_stack((X.ravel(), Y.ravel()))

    wedges = np.zeros(X.shape)
    counts = np.zeros(X.shape)
    results = np.zeros(X.shape)

    # Iterate over poinst in R^2 and determine sign patterns
    for idx, x in enumerate(grid):
        # Compute ReLU
        res = np.maximum(np.dot(W, x), 0)

        # Determine sign pattern of result
        signs = np.sign(res)

        # Convert the sign pattern to a wedge index
        wedge_index = np.sum((signs == 1) * (2 ** np.arange(len(signs))))
        
        wedges.ravel()[idx] = wedge_index
        counts.ravel()[idx] = np.count_nonzero(signs == 1)
        results.ravel()[idx] = np.linalg.norm(res, ord=1)

    
    # Plot the Wdges in R2
    plt.figure(figsize=(12, 12))
    contour_levels = np.arange(0, 2**m, 1)
    plt.contourf(X, Y, wedges, levels=contour_levels, cmap="viridis", alpha=0.5)     # Plots the Wedges
    plt.contour(X, Y, wedges, levels=contour_levels, colors="white", linewidths=1)   # Lines between wedges
    
    # Add labels for each wedge index outside the plot
    for wedge_index in range(int(np.max(wedges)) + 1):
        y, x = np.where(wedges == wedge_index) # Find the coordinates for the label
        if len(x) > 0:
            # Use the mean position of the wedge's coordinates for the label
            label_x = np.mean(X[y, x])
            label_y = np.mean(Y[y, x])
            
            x = np.array([label_x, label_y])
            res = np.maximum(np.dot(W, x), 0)
            signs = np.sign(res)
            signs[signs == -1] = 0

            color = "red" if np.sum(signs) < 2 else "white"
            plt.text(label_x + 0.1, label_y + 0.1, str(signs.astype(int)), fontsize=18, color=color, ha="center")

    # Plot the weights
    for i in range(m):
        plt.quiver(0, 0, W[i, 0], W[i, 1], angles="xy", scale_units="xy", scale=1)
        plt.text(W[i, 0] + 0.1, W[i, 1] + 0.1, f"$w_{i+1}$")

    plt.axhline(0, color="black", linewidth=1.5)  # Horizontal line at y=0
    plt.axvline(0, color="black", linewidth=1.5)  # Vertical line at x=0            
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Determine if Layer is Inyective
    verdict = "Not Injective" if 1. in np.unique(counts) else "Injective"
    plt.title(f"Wedge Visualization: Layer is {verdict}")
    
    save_path = os.path.join(os.getcwd(), "img", save_name+".png")
    plt.savefig(save_path)
    plt.close()

    
    # Plot the 3D plot
    norm = plt.Normalize(wedges.min(), wedges.max())
    colors = plt.cm.viridis(norm(wedges))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, results, facecolors=colors, shade=False)
    plt.title(f"Layer is {verdict}")
    
    save_path = os.path.join(os.getcwd(), "img", save_name+"_3D.png")
    plt.savefig(save_path)
    plt.close()




def main():
    parser = argparse.ArgumentParser(description="Visualize the weges created in a single ReLU layer")
    parser.add_argument("--mode", default="Paper", type=str, required=True, help="Either 'Paper', to reproduce the results of the paper or 'Random' to get the results of a random Weigt matrix")
    parser.add_argument("--num_weights", default=4, type=int, required=False, help="If mode was set to random this sets the numbers of vectors in the weight matrix")
    parser.add_argument("--check_inverse", default=True, type=bool, required=False, help="If mode was set to random this computes the weges of the random weights with adiitional inverted vectors")
    args = parser.parse_args()

    if args.mode == "Paper":
        # Figure 2 left from Paper
        W_1 = np.array([[ 2,    1],   # w_1
                        [-2,    2],   # w_2
                        [-1.5, -2],   # w_3
                        [ 2,   -1]])  # w_4
        visualize_wedges(W_1, "Paper_non-injectiv")

        # Figure 2 middle from Paper
        W_2 = np.array([[ 2,  1],   # w_1
                        [-2,  1],   # w_2
                        [-2, -1],   # w_3 = -w_1
                        [ 2, -1]])  # w_4 = -w_2
        visualize_wedges(W_2, "Paper_injectiv")
    
    elif args.mode == "Random":
        W_rand = 2 * np.random.rand(args.num_weights, 2) - 1
        visualize_wedges(W_rand, "Random_Weights")

        if args.check_inverse:
            W_rand_inv = -W_rand
            W_rand_comb = np.concatenate([W_rand, W_rand_inv])
            visualize_wedges(W_rand_comb, "Random_Weights_with_inverse")
    
    else: 
        raise NotImplementedError (f"Mode {args.mode} is not supported use one of ['Paper', 'Random']")



if __name__ == "__main__":
    main()
        


    




    
    