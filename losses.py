import torch
import torch.nn.functional as F

def relu_evidence(y):
    return F.relu(y)

def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood

def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div

def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div

def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step=10, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss

def reedl_loss(output, target, epoch_num, num_classes, annealing_step=10, device=None, lamb1=1.0, lamb2=0.8):
    if not device:
        device = output.device

    # : ModifiedEvidentialN.py:89
    evidence = F.softplus(output)

    #  compute_mse (ModifiedEvidentialN.py:143-151)
    actual_num_classes = evidence.shape[1]
    sum_evidence = torch.sum(evidence, dim=1, keepdim=True)
    projected_prob = (evidence + lamb2) / (
        evidence + lamb1 * (sum_evidence - evidence) + lamb2 * actual_num_classes
    )

    # target  one-hot  (64, 9)
    y = target.to(device)

    # : gap.pow(2).sum(-1).mean()
    gap = y - projected_prob
    loss = torch.mean(torch.sum(gap ** 2, dim=1))

    return loss

def edl_log_loss(output, target, epoch_num, num_classes, annealing_step=10, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss

def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step=10, device=None
):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1

    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss

def get_uncertainty(output, num_classes, lamb2=0.8):
    evidence = F.softplus(output)
    alpha = evidence + lamb2
    S = torch.sum(alpha, dim=1, keepdim=True)
    return (num_classes * lamb2) / S

def get_expectedprob(output, class_idx, lamb2=0.8):
    evidence = F.softplus(output)
    alpha = evidence + lamb2
    S = torch.sum(alpha, dim=1, keepdim=True)

    if isinstance(class_idx, torch.Tensor) and class_idx.dim() > 1:
        # one-hot encoding: return (batch, 1) weighted probability
        alpha_class = torch.sum(alpha * class_idx, dim=1, keepdim=True)
        return alpha_class / S
    elif isinstance(class_idx, torch.Tensor) and class_idx.dim() == 1:
        # 1D class indices: return (batch, n_class) full probability matrix
        return alpha / S
    else:
        # scalar class index: return (batch, 1) specific class probability
        alpha_class = alpha[:, class_idx].unsqueeze(1)
        return alpha_class / S
