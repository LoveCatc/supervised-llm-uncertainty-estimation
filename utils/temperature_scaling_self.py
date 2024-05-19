import torch
from torch import nn, optim



class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self):
        super(ModelWithTemperature, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.nll_criterion = nn.CrossEntropyLoss()
        # get the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
    def forward(self, logits):
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        # make sure the temperature is on the same device as the logits
        temperature = temperature.to(logits.device)

        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def get_temperature(self, logits, labels):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        if len(logits.shape) == 1 or logits.shape[1] == 1:
            self.nll_criterion = nn.BCEWithLogitsLoss()

        if self.device == torch.device("cuda"):
            self.cuda()
            self.nll_criterion.cuda()

        # First: collect all the logits and labels for the validation set

        logits.to(self.device)
        labels.to(self.device)

        
        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=100)

        def eval():
            optimizer.zero_grad()
            loss = self.nll_criterion(self.temperature_scale(logits.float()), labels.float())
            loss.backward()
            return loss
        optimizer.step(eval)
        temperature = self.temperature.item()

        return temperature



