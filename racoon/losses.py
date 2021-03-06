import torch
import torch.nn as nn

from config import CFG

# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        loss = nn.BCEWithLogitsLoss(reduction="mean")(preds, targets)
        return loss


class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")(preds, targets)
        probas = torch.sigmoid(preds)
        loss = (
            targets * self.alpha * (1.0 - probas) ** self.gamma * bce_loss
            + (1.0 - targets) * probas ** self.gamma * bce_loss
        )
        loss = loss.mean()
        return loss


class BCE2WayLoss(nn.Module):
    def __init__(self, weights=[1, 0.5]):
        super().__init__()
        self.bce = BCELoss()
        self.weights = weights

    def forward(self, input, target):
        input_ = input["logit"]
        target = target.float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.bce(input_, target)
        aux_loss = self.bce(clipwise_output_with_max, target)

        return self.weights[0] * loss + self.weights[1] * aux_loss


class BCEFocal2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = BCEFocalLoss()

        self.weights = weights

    def forward(self, input, target):
        input_ = input["logit"]
        target = target.float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.focal(input_, target)
        aux_loss = self.focal(clipwise_output_with_max, target)

        return self.weights[0] * loss + self.weights[1] * aux_loss


__CRITERIONS__ = {
    # "BCEFocalLoss": BCEFocalLoss,
    "BCEFocal2WayLoss": BCEFocal2WayLoss,
    # "BCELoss": BCELoss,
    "BCE2WayLoss": BCE2WayLoss,
}


def get_criterion():
    if hasattr(nn, CFG.loss_name):
        return nn.__getattribute__(CFG.loss_name)(**CFG.loss_params)
    elif __CRITERIONS__.get(CFG.loss_name) is not None:
        return __CRITERIONS__[CFG.loss_name](**CFG.loss_params)
    else:
        raise NotImplementedError
