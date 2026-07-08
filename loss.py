import torch
import torch.nn as nn


class NaNMSELoss():
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["modelname"]

    def fit(self, y_pred, y_true, lossmse):
        mask = y_true == y_true
        y_true = y_true[mask]
        y_pred = torch.squeeze(y_pred[mask])
        loss = torch.sqrt(lossmse(y_true, y_pred))
        return loss


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    loss = awl(2, 3)
    print(loss)
