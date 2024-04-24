import torch
from torch import nn, optim



class PlatScaling(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self,num_dim):
        super(PlatScaling, self).__init__()
        self.a = nn.Parameter(torch.ones(num_dim) * 1.0)
        self.b = nn.Parameter(torch.ones(num_dim) * 0.0)
        if num_dim>1:
            self.nll_criterion = nn.CrossEntropyLoss()
        else:
            self.nll_criterion = nn.BCEWithLogitsLoss()
        # get the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
    def forward(self, logits):
        return self.platt_scale(logits)

    def platt_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand a and b to match the size of logits
        if self.device==torch.device("cuda"):
            logits = logits.to(self.device)
            self.a.cuda()
            self.b.cuda()
        
        a = self.a.reshape(1,-1).expand(logits.size(0), -1)
        b = self.b.reshape(1,-1).expand(logits.size(0), -1)

        return a*logits + b

    # This function probably should live outside of this class, but whatever
    def get_a_and_b(self, logits, labels):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        if self.device == torch.device("cuda"):
            self.cuda()
            self.nll_criterion.cuda()
            self.a.cuda()
            self.b.cuda()

        # First: collect all the logits and labels for the validation set

        logits = logits.to(self.device)
        labels = labels.to(self.device)

        
        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.a,self.b], lr=0.1, max_iter=100)

        def eval():
            optimizer.zero_grad()
            loss = self.nll_criterion(self.platt_scale(logits.float()), labels.float())
            loss.backward()
            return loss
        optimizer.step(eval)
        a = self.a.cpu().detach()
        b = self.b.cpu().detach()

        return a,b



