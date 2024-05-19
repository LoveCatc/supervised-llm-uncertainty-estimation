from torch.nn import functional as F
from torch import nn
import torch

def calculate_ece(logits_input,labels_input):
    logits = logits_input.cpu()
    labels = labels_input.cpu()
    ece = _ECELoss()
    ece_value = ece(logits,labels).item()
    return ece_value

def calculate_nll(logits_input,labels_input):
    logits = logits_input.cpu()
    labels = labels_input.cpu()
    if len(logits.shape) >1 and logits.shape[1]>1:
        nll = nn.CrossEntropyLoss()
        nll_value = nll(logits.float(),labels).item()
    else:
        nll = nn.BCEWithLogitsLoss()
        # reshape logits and labels
        logits = logits.reshape(-1)
        labels = labels.float().reshape(-1)

        # avoid nan
        logits = logits.clamp(min=-1e6,max=1e6)

        nll_value = nll(logits.float(),labels.float()).item()
    return nll_value

def calculate_nll_ece(logits,labels):
    ece = calculate_ece(logits,labels)
    nll = calculate_nll(logits,labels)
    return nll,ece

def calculate_Brier(logits_input, labels_input):
    """
    Calculate the Brier score of a set of logits and labels
    logits (Tensor): a tensor of shape (N, C), where C is the number of classes
    labels (Tensor): a tensor of shape (N,), where each element is in [0, C-1]
    """
    logits = logits_input.cpu()
    labels = labels_input.cpu()
    if len(logits.shape) >1 and logits.shape[1]>1:
        # Convert labels to one-hot
        labels_one_hot = torch.zeros_like(logits)
        labels_one_hot.scatter_(1, labels.view(-1, 1), 1)

        softmaxes = F.softmax(logits, dim=1)
        
        # Calculate Brier score
        brier_score = ((softmaxes - labels_one_hot)**2).mean()
        return brier_score.item()
    else:
        labels = labels.float()
        brier_score = ((torch.sigmoid(logits) - labels)**2).mean()
        return brier_score.item()

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.bin_lowers[0] = self.bin_lowers[0] - 1e-6

    def forward(self, logits_input, labels_input):
        # move to cpu
        logits = logits_input.cpu()
        labels = labels_input.cpu()

        if len(logits.shape) >1 and logits.shape[1]>1:
            softmaxes = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(softmaxes, 1)
            accuracies = predictions.eq(labels)
        else:
            confidences = torch.sigmoid(logits)
            accuracies = labels
        
        
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
