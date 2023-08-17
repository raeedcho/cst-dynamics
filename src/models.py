import ssa
import numpy as np
import torch
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import TruncatedSVD,PCA

from . import dekodec

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

class JointSubspace(BaseEstimator,TransformerMixin):
    '''
    Model to find a joint subspace given multiple datasets in the same full-D space 
    and a number of dimensions to use for each dataset.

    This model will first reduce the dimensionality of the datasets to num_dims using TruncatedSVD,
    then concatenate the resulting PCs to form the joint subspace. Lastly, it will orthogonalize
    the joint subspace using SVD. The result will be a projection matrix from full-D space to
    the joint subspace (n_features x (num_dims*num_conditions)).

    Note: This class will not remove the mean of the datasets before performing TruncatedSVD by default.

    Arguments:
        df - (pd.DataFrame) DataFrame containing data (e.g. firing rates) and condition (e.g. task).
            Data will be grouped by the provided condition column to form multiple datasets.
            Each element of df[signal] is a numpy array with features along columns
            and optionally observations along rows. These arrays will be stacked via
            np.row_stack() to form a single data matrix for each dataset.
        signal - (str) name of column in df containing data
        condition - (str) name of column in df containing condition labels
        num_dims - (int) number of dimensions to use for each data matrix to compose
            the joint subspace.
        orthogonalize - (bool) whether or not to orthogonalize the joint subspace
            using SVD. Default is True.
        remove_latent_offsets - (bool) whether or not to remove the mean of each
            dataset before projecting into the joint subspace. If True, the mean of
            each resultant latent dimension will be 0. If False, the offsets in the
            original signals (signal means) will be passed through the transformation.
            Default is True, as in normal PCA.

    Returns:
        (numpy array) projection matrix from full-D space to joint subspace
            (n_features x (num_dims*num_conditions))
    '''
    def __init__(self,n_comps_per_cond=2,orthogonalize=True,signal=None,condition=None,remove_latent_offsets=True):
        '''
        Initiates JointSubspace model.
        '''
        assert signal is not None, "Must provide signal column name"
        assert condition is not None, "Must provide condition column name"

        self.n_comps_per_cond = n_comps_per_cond
        self.orthogonalize = orthogonalize
        self.signal = signal
        self.condition = condition
        self.remove_latent_offsets = remove_latent_offsets

    def fit(self,X,y=None):
        '''
        Fits the joint subspace model.

        Arguments:
            X - (pd.DataFrame) DataFrame containing data (e.g. firing rates) and condition (e.g. task
                Data will be grouped by the provided condition column to form multiple datasets.
            y - unused

        Returns:
            self - the fitted transformer object
        '''

        # group data by condition
        self.conditions_ = np.unique(X[self.condition])
        self.n_conditions_ = len(self.conditions_)
        self.n_components_ = self.n_comps_per_cond*self.n_conditions_
        self.full_mean_ = np.mean(np.row_stack(X[self.signal]),axis=0)
        self.cond_means_ = (
            X
            .groupby(self.condition)
            [self.signal]
            .apply(lambda x: np.mean(np.row_stack(x)-self.full_mean_,axis=0))
        )

        dim_red_models = (
            X
            .groupby(self.condition)
            [self.signal]
            .apply(lambda x: PCA(n_components=self.n_comps_per_cond).fit(np.row_stack(x)))
        )

        proj_mat = np.row_stack([model.components_ for model in dim_red_models])
        if not self.orthogonalize:
            self.P_ = proj_mat.T
        else:
            _,_,Vt = np.linalg.svd(proj_mat,full_matrices=False)
            self.P_ = dekodec.max_var_rotate(Vt.T,np.row_stack(X[self.signal]))

        return self

    def transform(self,X):
        '''
        Projects data into joint subspace.

        Arguments:
            X - (pd.DataFrame)
                DataFrame containing data (e.g. firing rates) and condition (e.g. task)

        Returns:
            (pd.DataFrame) New DataFrame with an additional column containing the
                projected data (column name is f'{self.signal}_joint_pca')
        '''
        assert hasattr(self,'P_'), "Model not yet fitted"

        if self.remove_latent_offsets:
            return (
                X
                .join(self.cond_means_,on=self.condition,rsuffix='_mean')
                .assign(**{
                    f'centered_{self.signal}': lambda df: df.apply(lambda s: s[self.signal] - self.full_mean_ - s[f'{self.signal}_mean'],axis=1),
                    f'{self.signal}_joint_pca': lambda df: df.apply(lambda s: np.dot(s[f'centered_{self.signal}'],self.P_),axis=1),
                })
                .drop(columns=[f'{self.signal}_mean',f'centered_{self.signal}'])
            )
        else:
            return (
                X
                .assign(**{
                    f'{self.signal}_joint_pca': lambda df: df.apply(lambda s: np.dot(s[self.signal],self.P_),axis=1),
                })
            )

class DekODec(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            var_cutoff=0.99,
            signal=None,
            condition=None,
        ):
        assert signal is not None, "Must provide signal column name"
        assert condition is not None, "Must provide condition column name"

        self.var_cutoff = var_cutoff
        self.signal = signal
        self.condition = condition

    def fit(self, X, y=None):
        X_conds_dict = (
            X
            .groupby(self.condition)
            [self.signal]
            .agg(np.row_stack)
            .to_dict()
        )

        self.subspaces = dekodec.fit_dekodec(X_conds_dict,var_cutoff=self.var_cutoff)

        return self

    def transform(self,X):
        '''
        Projects data into unique and shared subspaces.

        Arguments:
            X - (pd.DataFrame)
                DataFrame containing data (e.g. firing rates) and condition (e.g. task)

        Returns:
            (pd.DataFrame) New DataFrame with an additional column containing the
                projected data (column names are f'{self.signal}_{subspace_name}')
        '''
        assert hasattr(self,'subspaces'), "Model not yet fitted"

        return (
            X
            .assign(**{
                f'{self.signal}_{subspace_name}':
                    lambda df, proj_mat=proj_mat: df[self.signal].apply(
                        lambda arr: arr @ proj_mat
                    )
                for subspace_name,proj_mat in self.subspaces.items()
            })
        )