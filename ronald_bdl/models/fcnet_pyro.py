import torch
import torch.nn as nn
import pyro
from pyro.distributions import Normal, Uniform

from .fcnet import FCNet

class FCNetPyro(FCNet):
    def __init__(self, input_dim, output_dim, hidden_dim, n_hidden):
        super(FCNetPyro, self).__init__(input_dim, output_dim, hidden_dim, 0)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden

    def model(self, X, y):
        input_weight_prior = Normal(
            loc=torch.zeros_like(self.input[0].weight),
            scale=torch.ones_like(self.input[0].weight))

        input_bias_prior = Normal(
            loc=torch.zeros_like(self.input[0].bias),
            scale=torch.ones_like(self.input[0].bias))
        
        output_weight_prior = Normal(
            loc=torch.zeros_like(self.output.weight),
            scale=torch.ones_like(self.output.weight))

        output_bias_prior =  Normal(
            loc=torch.zeros_like(self.output.bias),
            scale=torch.ones_like(self.output.bias))

        priors = {
            'input.weight': input_weight_prior,
            'input.bias': input_bias_prior,
            'output.weight': output_weight_prior, 
            'output.bias': output_bias_prior
        }

        scale = pyro.sample("sigma", Uniform(0., 10.))

        lifted_module = pyro.random_module("module", self, priors)

        lifted_reg_model = lifted_module()

        # run the regressor forward conditioned on inputs
        prediction_mean = lifted_reg_model(X).squeeze()
        pyro.sample("obs",
                    Normal(prediction_mean, scale),
                    obs=y.squeeze())

    def guide(self, X, y):
        # First layer weight distribution priors
        input_weight_mu = torch.randn_like(self.input[0].weight)
        input_weight_sigma = torch.randn_like(self.input[0].weight)

        input_weight_mu_param = pyro.param("input_weight_mu", input_weight_mu)
        input_weight_sigma_param = pyro.param("input_weight_sigma", input_weight_sigma)

        input_weight_prior = Normal(loc=input_weight_mu_param, scale=input_weight_sigma_param)
    
        # First layer bias distribution priors
        input_bias_mu = torch.randn_like(self.input[0].bias)
        input_bias_sigma = torch.randn_like(self.input[0].bias)

        input_bias_mu_param = pyro.param("input_bias_mu", input_bias_mu)
        input_bias_sigma_param = pyro.param("input_bias_sigma", input_bias_sigma)

        input_bias_prior = Normal(loc=input_bias_mu_param, scale=input_bias_sigma_param)

        # Output layer weight distribution priors
        output_weight_mu = torch.randn_like(self.output.weight)
        output_weight_sigma = torch.randn_like(self.output.weight)

        output_weight_mu_param = pyro.param("output_weight_mu", output_weight_mu)
        output_weight_sigma_param = pyro.param("output_weight_sigma", output_weight_sigma)

        output_weight_prior = Normal(loc=output_weight_mu_param, scale=output_weight_sigma_param).independent(1)

        # Output layer bias distribution priors
        output_bias_mu = torch.randn_like(self.output.bias)
        output_bias_sigma = torch.randn_like(self.output.bias)

        output_bias_mu_param = pyro.param("output_bias_mu", output_bias_mu)
        output_bias_sigma_param = pyro.param("output_bias_sigma", output_bias_sigma)

        output_bias_prior = Normal(loc=output_bias_mu_param, scale=output_bias_sigma_param)

        priors = {
            'input.weight': input_weight_prior,
            'input.bias': input_bias_prior,
            'output.weight': output_weight_prior,
            'output.bias': output_bias_prior}
        
        lifted_module = pyro.random_module("module", self, priors)
    
        return lifted_module()

    def predict_dist(self, X_test, n_predictions, **kwargs):
        sampled_models = [self.guide(None, None) for _ in range(n_predictions)]
        yhats = [model(X_test).data for model in sampled_models]
        predictions = torch.stack(yhats)

        mean = torch.mean(predictions, 0)
        var = torch.var(predictions, 0)

        # If y_test is given, calculate RMSE and test log-likelihood
        metrics = {}

        if 'y_test' in kwargs:
            y_test = kwargs['y_test']
            reg_strength = torch.tensor(kwargs['reg_strength'], dtype=torch.float)

            # RMSE
            metrics['rmse_mc'] = torch.sqrt(torch.mean(torch.pow(y_test - mean, 2)))

            # RMSE (Non-MC)
            prediction_non_mc = self.forward(X_test)
            metrics['rmse_non_mc'] = torch.sqrt(torch.mean(torch.pow(y_test - prediction_non_mc, 2)))

            # test log-likelihood
            metrics['test_ll_mc'] = torch.mean(
                torch.logsumexp(- torch.tensor(0.5) * reg_strength * torch.pow(y_test[None] - predictions, 2), 0)
                - torch.log(torch.tensor(n_predictions, dtype=torch.float))
                - torch.tensor(0.5) * torch.log(torch.tensor(2 * np.pi, dtype=torch.float)) 
                + torch.tensor(0.5) * torch.log(reg_strength)
            )

        return predictions, mean, var, metrics
