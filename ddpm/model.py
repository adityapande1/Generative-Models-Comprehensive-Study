import torch
import pytorch_lightning as pl

# Additional imports
from torch import nn
import torch.nn.functional as F
import numpy as np

class LitDiffusionModel(pl.LightningModule):
    def __init__(self, n_dim=3, n_steps=200, lbeta=1e-5, ubeta=1e-2):
        super().__init__()
        """
        If you include more hyperparams (e.g. `n_layers`), be sure to add that to `argparse` from `train.py`.
        Also, manually make sure that this new hyperparameter is being saved in `hparams.yaml`.
        """
        self.save_hyperparameters()

        """
        Your model implementation starts here. We have separate learnable modules for `time_embed` and `model`.
        You may choose a different architecture altogether. Feel free to explore what works best for you.
        If your architecture is just a sequence of `torch.nn.XXX` layers, using `torch.nn.Sequential` will be easier.
        
        `time_embed` can be learned or a fixed function based on the insights you get from visualizing the data.
        If your `model` is different for different datasets, you can use a hyperparameter to switch between them.
        Make sure that your hyperparameter behaves as expecte and is being saved correctly in `hparams.yaml`.
        """
        self.time_embed = None
        self.model = nn.Sequential(nn.Linear(n_dim+4, 32),nn.ReLU(),nn.Linear(32, 32),nn.ReLU(),nn.Linear(32, 32),nn.ReLU(), nn.Linear(32, n_dim))

        """
        Be sure to save at least these 2 parameters in the model instance.
        """

        self.n_steps = n_steps
        self.n_dim = n_dim

        """
        Sets up variables for noise schedule
        """
        self.beta = None 
        self.alpha = None
        self.alpha_bar = [torch.tensor(0)]
        self.init_alpha_beta_schedule(lbeta, ubeta)
    

    def time_embed_single(self,t):
        """
        Returns positional time embedding of t using sin and cosine
        t : 0 order tensor ie scalar
        d : is the length of vector (here :4) 
        """
            
        
        d = 4   # dimension of time embedding
        vec = [] 
        for i in range(int(d/2)):
                den = 10000**(2*i/d)
                vec.append(np.sin(t/den))
                vec.append(np.cos(t/den))
        return torch.tensor(vec)




    def forward(self, x, t):
        """
        Similar to `forward` function in `nn.Module`. 
        Notice here that `x` and `t` are passed separately. If you are using an architecture that combines
        `x` and `t` in a different way, modify this function appropriately.
        """
       
        if not isinstance(t, torch.Tensor):
            t = torch.LongTensor([t]).expand(x.size(0))
        
        t_embed = []
        for time in t:
            embed = self.time_embed_single(time)
            t_embed.append(embed.tolist())
        t_embed = torch.tensor(t_embed)

        return self.model(torch.cat((x, t_embed), dim=1).float())

    def init_alpha_beta_schedule(self, lbeta, ubeta):
        """
        Set up your noise schedule. You can perhaps have an additional hyperparameter that allows you to
        switch between various schedules for answering q4 in depth. Make sure that this hyperparameter 
        is included correctly while saving and loading your checkpoints.
        """
        
        ns = 6

        if ns == 1:

            # Noise Schedule 1 
            self.beta = torch.linspace(lbeta, ubeta, self.n_steps + 1)
        
        elif ns == 2:
            # Noise Schedule 2 : Quadratic 
            t_by_T_sq = (torch.arange(0,self.n_steps + 1)/(self.n_steps))**2
            self.beta = lbeta + (ubeta - lbeta)*t_by_T_sq
        
        elif ns == 3:
            # Noise Schedule 3 : Cubic
            t_by_T_sq = (torch.arange(0,self.n_steps + 1)/(self.n_steps))**3
            self.beta = lbeta + (ubeta - lbeta)*t_by_T_sq

        elif ns == 4:
            # Noise Schedule 4 : logarithmic
            e = torch.exp(torch.tensor(1))
            a = (torch.exp(torch.tensor(ubeta/lbeta)) - e)/self.n_steps
            t_range = torch.arange(0,self.n_steps + 1)
            self.beta = lbeta*torch.log(a*t_range + e)
        
        elif ns == 5:
            # Noise Schedule 5 : Exponential
            t_by_T = (torch.arange(0,self.n_steps + 1)/(self.n_steps))
            exponent = t_by_T*torch.log(torch.tensor(ubeta/lbeta))
            self.beta = lbeta*torch.exp(exponent)
        
        elif ns == 6:
            # Noise Schedule 6 : Sinosoidal Simple
            t_by_T = (torch.arange(0,self.n_steps + 1)/(self.n_steps))
            self.beta = lbeta + (ubeta-lbeta)*torch.sin((torch.pi/2)*t_by_T)
        elif ns == 7:
            # Noise Schedule 7 : Cosine Simple
            t_by_T = (torch.arange(0,self.n_steps + 1)/(self.n_steps))
            self.beta = lbeta + (ubeta-lbeta)*torch.cos((torch.pi/2) - (torch.pi/2)*t_by_T)
        elif ns == 8:
            # Noise Schedule 8 : Linear plus Sinosoidal
            t_by_T = (torch.arange(0,self.n_steps + 1)/(self.n_steps))
            self.beta =  lbeta + (ubeta-lbeta)*t_by_T + torch.sin((torch.pi*2)*t_by_T)
        elif ns == 9:
            # Noise Schedule 9 : Cosine implemented in paper
            # Paper Link : https://arxiv.org/pdf/2102.09672.pdf
            pass




        self.alpha = 1 - self.beta
        
        (self.alpha_bar).append(self.alpha[1].item())
        for t in range(2, self.n_steps + 1):
            
            app = self.alpha_bar[t-1]*self.alpha[t].item() 
            (self.alpha_bar).append(app)
        
        self.alpha_bar = torch.tensor(self.alpha_bar)

    def q_sample(self, x, t):
        """
        Sample from q given x_t.
        t : should be scalar
        """
        eps = torch.randn_like(x)
        vec = torch.sqrt(1-self.beta[t+1])*x + torch.sqrt(self.beta[t+1])*eps
        return vec

    def p_sample(self, x, t):
        """
        Sample from p given x_t.
        t : should be scalar
        """
        eps_theta = self.forward(x,t)       # Find noise from the NNet
        alpha_bar_t = torch.prod(1-self.beta[1:t+1]) # Multiply 1-betas from 1 to t

        x_dash = ((self.beta[t])/(torch.sqrt(1-alpha_bar_t)))*eps_theta

        mu_theta = (x-x_dash)*(1/(1-self.beta[t]))

        eps = torch.randn_like(x)   # Sample from N(0,1)

        vec = mu_theta + torch.sqrt(self.beta[t])*eps   # Reparameterize

        return vec

    def training_step(self, batch, batch_idx):
        """
        Implements one training step.
        Given a batch of samples (n_samples, n_dim) from the distribution you must calculate the loss
        for this batch. Simply return this loss from this function so that PyTorch Lightning will 
        automatically do the backprop for you. 
        Refer to the DDPM paper [1] for more details about equations that you need to implement for
        calculating loss. Make sure that all the operations preserve gradients for proper backprop.
        Refer to PyTorch Lightning documentation [2,3] for more details about how the automatic backprop 
        will update the parameters based on the loss you return from this function.

        References:
        [1]: https://arxiv.org/abs/2006.11239
        [2]: https://pytorch-lightning.readthedocs.io/en/stable/
        [3]: https://www.pytorchlightning.ai/tutorials
        """
        epsMatrix = []        # will store epsilons
        inputMatrix = []      # will store the concatenated inputs
        
        for x_not in batch:
    
            t  = torch.randint(1, self.n_steps, (1,))  # Sample t from uniform
            t = t.item()                          # Numerical value of t
            eps =  torch.randn_like(x_not)        # Sample from N(0,I) the noise

            a = torch.sqrt(self.alpha_bar[t])*x_not + torch.sqrt(1-self.alpha_bar[t])*eps
            #b will later be the time embedding for now it's 4d
            #b = torch.tensor([1.22,3.44,2.44,1.55])
            b = self.time_embed_single(t)
            c = torch.cat((a,b), dim=0)

            inputMatrix.append(c.tolist())
            epsMatrix.append(eps.tolist())
        
        epsMatrix = torch.tensor(epsMatrix)
        epsMatrix = epsMatrix.float()
        inputMatrix = torch.tensor(inputMatrix)
        inputMatrix = inputMatrix.float()
        
        #print(inputMatrix)
        
        
        x_hat = self.model(inputMatrix)
        loss = F.mse_loss(x_hat, epsMatrix)
        return loss
    

    def sample(self, n_samples, progress=False, return_intermediate=False):
        """
        Implements inference step for the DDPM.
        `progress` is an optional flag to implement -- it should just show the current step in diffusion
        reverse process.
        If `return_intermediate` is `False`,
            the function returns a `n_samples` sampled from the learned DDPM
            i.e. a Tensor of size (n_samples, n_dim).
            Return: (n_samples, n_dim)(final result from diffusion)
        Else
            the function returns all the intermediate steps in the diffusion process as well 
            i.e. a Tensor of size (n_samples, n_dim) and a list of `self.n_steps` Tensors of size (n_samples, n_dim) each.
            Return: (n_samples, n_dim)(final result), [(n_samples, n_dim)(intermediate) x n_steps]
        """
        
        x_T = torch.randn(n_samples, self.n_dim)

        myList = [] 
        
        for t in range(self.n_steps, 0, -1) :

            if return_intermediate == True: 
                myList.append(x_T)


            # Define Different Factors
            fact_1 = 1/torch.sqrt(self.alpha[t])
            fact_2 = (1-self.alpha[t])/torch.sqrt(1-self.alpha_bar[t])
            fact_3 = torch.sqrt(self.beta[t])

            # Sample Z's 
            z = torch.randn(n_samples, self.n_dim)

            # Calculate x_t-1
            x_T = fact_1*(x_T - fact_2*self.forward(x_T, t)) + fact_3*z

        #myList.reverse()

        if return_intermediate == True: 
            result = x_T, myList
        else:
            result = x_T
            
        return result

    def configure_optimizers(self):
        """
        Sets up the optimizer to be used for backprop.
        Must return a `torch.optim.XXX` instance.
        You may choose to add certain hyperparameters of the optimizers to the `train.py` as well.
        In our experiments, we chose one good value of optimizer hyperparameters for all experiments.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
