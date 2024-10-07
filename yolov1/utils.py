import torch

from config import settings


def get_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate iou for prediction versus ground truth

    Params:
        pred: prediction of shape (N, 7, 7, B)
        target: ground truth of shape (N, 7, 7, B)

    Output:
        ious: shape (N, 7, 7, B, B)

    Notes:
        The iou is calculated on each prediction and each ground truth
        bounding box. Suppose there're two predictions p1, p2 and two
        ground truth boxes t1, t2. The output will be
        [[iou(p1, t1), iou(p1, t2)], [iou(p2, t1), iou(p2, t2)]].
    """
    # Each of shape (N, S, S, B, 2)
    ptl, pbr = get_coords(pred)
    ttl, tbr = get_coords(target)

    # Grid comparison (N, S, S, B, B, 2)
    _expand = (-1, -1, -1, settings.B, settings.B, 2)
    tl = torch.max(ptl.unsqueeze(4).expand(_expand), ttl.unsqueeze(3).expand(_expand))
    br = torch.min(pbr.unsqueeze(4).expand(_expand), tbr.unsqueeze(3).expand(_expand))

    # Clamp in case of 0 division
    intersection = torch.clamp(br - tl, min=0.0)

    # Shape (N, S, S, B, B)
    area_inter = intersection[..., 0] * intersection[..., 1]
    area_predict = (pbr - ptl)[..., 0:1] * (pbr - ptl)[..., 1:2]
    area_target = (tbr - ttl)[..., 0:1] * (tbr - ttl)[..., 1:2]
    area_union = area_predict + area_target - area_inter

    return area_inter / (area_union + 1e-6)


def get_coords(box: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return top left and bottom right coordinates of the given box

    Params:
        box: shape (N, S, S, B * 5 + C)

    Output:
        coordinates: two tensors, both of shape (N, S, S, B, 2).

    Notes:
        Each tensor in the tuple is of shape (N, S, S, B, 2) in which the
        last dimension represents [x, y]; B represents the number of boxes.
        Overall, the last dimension can be abstracted as [[x1,y1],[x2,y2],
        ...,[xb, yb]].
    """
    x, y, w, h = (
        get_bbox_attr(box, "x"),
        get_bbox_attr(box, "y"),
        get_bbox_attr(box, "w"),
        get_bbox_attr(box, "h"),
    )

    x_tl = x - w / 2
    y_tl = y - h / 2
    x_br = x + w / 2
    y_br = y + h / 2

    return (torch.stack((x_tl, y_tl), dim=4), torch.stack((x_br, y_br), dim=4))


def get_bbox_attr(box: torch.Tensor, attr: str) -> torch.Tensor:
    """
    Get bounding box attribute

    Params:
        box: bounding box of shape (N, S, S, B * 5 + 2)
        attr: the attribute to require, value range: {"x", "y", "w", "h", "c"}

    Output:
        attributes: shape (N, S, S, B)

    Notes:
        The last dimension is shaped in the following format.
        Format: [b1_x, b1_y, b1_w, b1_h, b1_c, b2_x, ..., b2_h, C1, C2, ..., C20]
        B number of (default to 2) predicted bounding boxes containing five attributes,
        respectively being x, y, w, h, and c, followed by C number of class attributes
        (which class this object is).
    """

    _map = {"x": 0, "y": 1, "w": 2, "h": 3, "c": 4}

    if attr not in _map:
        raise ValueError(f"{attr} not int attribute set {list(_map.keys())}.")
    attr = _map[attr]

    return box[..., attr : settings.B * 5 : 5]


pred = torch.randn(5, 7, 7, 30)
target = torch.randn(5, 7, 7, 30)
