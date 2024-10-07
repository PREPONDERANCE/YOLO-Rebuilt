import torch
import torch.nn.functional as F
from torch import nn

from utils import get_iou, get_bbox_attr, get_classes


class SSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def _sse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Perform Sum Squared Error on `pred` and `target`

        Params:
            pred: predictions of shape (N, S, S, B)
            target: ground truth of shape (N, S, S, B)

        Output:
            sse: the result of sum squared error loss
        """

        pred = pred.flatten(end_dim=-2)
        target = target.flatten(end_dim=-2).expand_as(pred)

        return F.mse_loss(pred, target, reduction="sum")

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Params:
            pred: output predictions of shape (N, S, S, B * 5 + C)
            target: ground truth of shape (N, S, S, B * 5 + C)

        Output:
            Scalar value indicating loss

        Reference:
            https://arxiv.org/pdf/1506.02640

        Notes:
            The identity function returns 1 if there's a bounding box predicted
            in that cell and this box is "responsible" for this cell. (Owns the
            most IOU score out of the 2 bounding box [B=2 in YOLOv1])
        """

        iou = get_iou(pred, target)  # Shape (N, S, S, B, B)
        iou_max = torch.max(iou, dim=-1).values  # Shape (N, S, S, B)

        responsible = torch.zeros_like(pred[..., 0:1])
        responsible[torch.argmax(iou_max, dim=-1, keepdim=True)] = 1
        obj_present = (get_bbox_attr(target, attr="c") > 0)[..., 0:1]

        obj = obj_present * responsible
        noobj = ~(obj.bool())

        x_pred, x_tar = obj * get_bbox_attr(pred, attr="x"), obj * get_bbox_attr(target, attr="x")
        x_loss = self.lambda_coord * self._sse(x_pred, x_tar)

        y_pred, y_tar = obj * get_bbox_attr(pred, attr="y"), obj * get_bbox_attr(target, attr="y")
        y_loss = self.lambda_coord * self._sse(y_pred, y_tar)

        w_pred, w_tar = obj * get_bbox_attr(pred, attr="w"), obj * get_bbox_attr(target, attr="w")
        h_pred, h_tar = obj * get_bbox_attr(pred, attr="h"), obj * get_bbox_attr(target, attr="h")
        w_pred = torch.sign(w_pred) * torch.sqrt(torch.abs(w_pred))
        h_pred = torch.sign(h_pred) * torch.sqrt(torch.abs(h_pred))

        w_loss = self.lambda_coord * self._sse(w_pred, w_tar)
        h_loss = self.lambda_coord * self._sse(h_pred, h_tar)

        c_pred, c_tar = get_bbox_attr(pred, attr="c"), get_bbox_attr(target, attr="c")
        c_obj_loss = self._sse(obj * c_pred, obj * c_tar)
        c_noobj_loss = self.lambda_noobj * self._sse(noobj * c_pred, noobj * c_tar)

        cls_pred, cls_tar = obj * get_classes(pred), obj * get_classes(target)
        cls_loss = self._sse(cls_pred, cls_tar)

        return x_loss + y_loss + w_loss + h_loss + c_obj_loss + c_noobj_loss + cls_loss
