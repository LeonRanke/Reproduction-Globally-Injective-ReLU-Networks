# Reproduce Figure from [Globally Injective ReLU Networks](https://arxiv.org/abs/2006.08464)

## Description

This repo reproduces the results of Figure 2 in the paper [Globally Injective ReLU Networks](https://arxiv.org/abs/2006.08464).

Given a weight matrix $W$ ist test wether the layer $ReLU(W x)$ is Injective or not and visualizes the wedges crated from the RelU layer.

<div style="display: flex; justify-content: space-around;">
    <img src="img/Paper_non-injectiv.png" alt="Image 1" width="300"/>
    <img src="img/Paper_injectiv.png" alt="Image 2" width="300"/>
    <img src="img/Paper_non-injectiv_3D.png" alt="Image 3" width="300"/>
</div>




## Installation

1. Create a virtual environment: `python3 -m venv [venv_name]`
2. Activate the virtual environment: `source [venv_name]\bin\activate`
3. Upgrade pip: `pip install --upgrade pip`
4. Install dependencies: `pip install -r requirements.txt`

## Usage

**Run the script:** `python3 wedge_visualization.py --mode=Paper`

* With `mode=Paper` the script will reproduce Figure 2 from the paper
* Set `mode=Random` to visualize the wedges of a random weight matrix, in this mode `--num_weights` can be set as the number of vectors of the weight matrix (high values will result in a high computational time). Also if `check_inverse` is set to `True` the wedges of the same random weight matrix including the inverse of all of its vectors will be shown. This guarantees inactivity. (I recommend setting this to `False` if you are using a large number of vectors to avoid computational overhead)

The images will be stored in the `img` folder.

## 

The [Globally Injective ReLU Networks](https://arxiv.org/abs/2006.08464) paper:
```
@misc{puthawala2021globallyinjectiverelunetworks,
      title={Globally Injective ReLU Networks}, 
      author={Michael Puthawala and Konik Kothari and Matti Lassas and Ivan Dokmanić and Maarten de Hoop},
      year={2021},
      eprint={2006.08464},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2006.08464}, 
}
```