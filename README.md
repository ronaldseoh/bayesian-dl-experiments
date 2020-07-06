# bayesian-dl-experiments

This repository contains the codes used to produce the results from the technical report [*Qualitative Analysis of Monte Carlo Dropout*](https://link.iamblogger.net/mc-dropout-qual-report).

Nearly all the results were produced with [PyTorch](https://link.iamblogger.net/pytorch) codes in this repo and [`ronald_bdl` repository](https://link.iamblogger.net/ronald-bdl-py), except for Figure 5, Table 1 and Table 2, which were done with [the codes from Gal and Ghahramani 2016](https://link.iamblogger.net/bdlexperiments).

[`ronald_bdl`](https://link.iamblogger.net/ronald-bdl-py) needs to be installed as a Python package before running the notebooks. This package contains pre-defined PyTorch NN models (`ronald_bdl.models`) and [Dataset](https://link.iamblogger.net/pytorch-data-tutorial) classes (`ronald_bdl.datasets`). Please run the following command using `pip`:

```bash
pip install git+https://github.com/ronaldseoh/ronald_bdl.git
```

If you want to modify the code within `ronald_bdl`, please clone/download the `ronald_bdl` repo, apply your changes, and install your version using the command `pip install .`

Please refer to the descriptions below for what each Jupyter notebook does:

- `experiment_comparison_toy.ipynb`: This notebook was created to produce the results in the section 3.1: "Uncertainty Information" of the report where we wanted to visually analyze how tuning the parameters changes the predictive distribution captured by MC dropout, when trained on toy datasets where we define the actual data generating function and noise.
- `experiment_error_convergence_{1_uci_fcnet, 2_cifar10_simplecnn}.ipynb`: Used to produce the result in the section 3.2: "Improvements in Predictive Performance and the section 4.1: "Number of Training Epochs". `1_uci_fcnet` trains a fully-connected network with the UCI datasets originally used in Gal and Ghahramani 2016, and `2_cifar10_simplecnn` trains a simple convolution NN with the [CIFAR-10](https://link.iamblogger.net/cifar10-pytorch) dataset.
- `experiment_number_of_test_predictions_{1_uci_fcnet, 2_cifar10_simplecnn}.ipynb`: Used to produce the result in the section 4.2: "Number of Test Predictions".

**Note**: While there are some references to [`Pyro`](https://link.iamblogger.net/pyro) in the code as we originally intended to implement a BNN using MCMC for comparison, The results using HMC are currently not included in the report due to some technical issues.

# License

`bayesian-dl-experiments` is licensed under MIT license. Please check `LICENSE`.