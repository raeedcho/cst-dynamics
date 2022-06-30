import ssa
import numpy as np
import torch


class SSA(object):
    def __init__(
        self,
        R=None,
        lam_sparse=None,
        lr=0.001,
        n_epochs=3000,
        orth=True,
        lam_orthog=None,
        scheduler_params_input=dict(),
        center_data=True,
    ):
        self.ssa_params = {
            "R": R,
            "lam_sparse": lam_sparse,
            "lr": lr,
            "n_epochs": n_epochs,
            "orth": orth,
            "lam_orthog": lam_orthog,
            "scheduler_params_input": scheduler_params_input,
        }
        self.center_data = center_data
        self.latent = None
        self.losses = None
        self.model = None

        # if self.ssa_params["orth"]:
        #     self.model = ssa.models.LowROrth()

    def fit(self, X, sample_weight=None):
        if self.center_data:
            X_data = np.copy(X - np.mean(X, axis=0))
        else:
            X_data = np.copy(X)

        self.model, self.latent, _, self.losses = ssa.fit_ssa(
            X_data, sample_weight=sample_weight, **self.ssa_params
        )

    def transform(self, X):
        if self.center_data:
            X_data = np.copy(X - np.mean(X, axis=0))
        else:
            X_data = np.copy(X)

        [X_torch] = ssa.torchify([X_data])
        latent, _ = self.model(X_torch)

        return latent.cpu().detach().numpy()

    def save_model_to_file(self, filepath):
        '''
        Saves model to a file (to be loaded by `load_model_from_file`)
        in the form of a state_dict
        
        Parameters
        ----------
        filepath : str
            Path to the file to save the model to
            
        Returns
        -------
        None
        '''
        state_dict = self.model.state_dict()
        torch.save(state_dict, filepath)

    def load_model_from_file(self, filepath):
        '''
        Loads the model from a file (saved by `save_model_to_file`)
        
        Parameters
        ----------
        filepath : str
            Path to the file containing the model
            
        Returns
        -------
        None
        '''
        assert self.model is None, "Model already loaded"

        state_dict = torch.load(filepath)
        U_init = state_dict['fc1.weight'].clone().detach().cpu().numpy()
        b_init = state_dict['fc2.bias'].clone().detach().cpu().numpy()
        input_size = U_init.shape[1]
        output_size = state_dict['fc2.parametrizations.weight.original'].shape[0]
        self.ssa_params['R'] = U_init.shape[0]
        
        if self.ssa_params["orth"]:
            self.model = ssa.models.LowROrth(input_size, output_size, self.ssa_params['R'],U_init,b_init)
        else:
            self.model = ssa.models.LowRNorm(input_size, output_size, self.ssa_params['R'],U_init,b_init)

        self.model.load_state_dict(state_dict)
        
        # use GPU if available
        if torch.cuda.is_available():
            self.model.cuda()

        self.model.eval()
