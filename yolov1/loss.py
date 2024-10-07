import torch

from torch import nn


class SSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Params:
            pred: output predictions of shape (N, 7, 7, 30)
            target: ground truth of shape (N, 7, 7, 30)

        Output:
            Scalar value indicating loss

        Reference:
            https://arxiv.org/pdf/1506.02640

        Notes:
            The identity function returns 1 if there's a bounding box predicted
            in that cell and this box is "responsible" for this cell. (Owns the
            most IOU score out of the 2 bounding box [B=2 in YOLOv1])
        """

        pass
