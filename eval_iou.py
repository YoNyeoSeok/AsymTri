import os.path as osp
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import pandas as pd


def compute_threshold_table(
        max_output, argmax_output,
        C=19, N=10, ignore_class=False):
    """
        max_output: Bx(WH|WxH)
        argmax_output: Bx(WH|WxH)
        if ignore_class- is None:
            return BxCxN
        else
            return Bx1xN
    """
    assert max_output.shape == argmax_output.shape
    if max_output.dim() == 2:
        B, WH = max_output.shape
    elif max_output.dim() == 3:
        B, W, H = max_output.shape
        WH = W*H
    else:
        assert False, "output dimension should be 2 or 3. {}".format(
            max_output.dim())
    # BxWH
    sorted_max_output, sorted_idx_max_output = max_output.reshape(
        B, -1).sort(-1)
    sorted_idx_argmax_output = argmax_output.reshape(
        B, -1).gather(1, sorted_idx_max_output)
    # BxWH+1
    inf_pad_max_output = torch.cat(
        [sorted_max_output,
         torch.full_like(sorted_max_output[:, :1], float('inf'))],
        dim=1)

    # BxWHx(1|C)
    mask = sorted_idx_argmax_output.unsqueeze(-1)
    mask = mask != C + \
        1 if ignore_class else mask == torch.arange(C, device=mask.device)

    # BxNx(1|C)
    threshold_lin_idx = torch.from_numpy(
        # BxNx(1|C)
        np.linspace(0, mask.sum(1).cpu().numpy(), N+1, axis=1)[:, :-1]
    ).to(mask.device).round().long()

    # BxWH+1x(1|C)
    last_pad_mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
    true_idx_sorted = (torch.arange(WH+1, device=mask.device).unsqueeze(-1) -
                       last_pad_mask.int()*WH).argsort(1)
    # [BxWH+1x(1|C)].gather(1, BxNx(C|1)) -> BxNx(C|1)
    threshold_output_idx = true_idx_sorted.gather(
        1, threshold_lin_idx)

    # [BxWH+1].gather(1, BxNx(C|1) -> BxN(C|1) ) -> BxN(C|1)
    per_class_threshold_output = inf_pad_max_output.gather(
        1, threshold_output_idx.reshape(B, -1))
    # BxN(C|1) -> BxNx(C|1)
    del mask
    del threshold_lin_idx
    del last_pad_mask
    del true_idx_sorted
    return per_class_threshold_output.reshape_as(threshold_output_idx)


def per_class_filter_two_output(
        max_output1, max_output2,
        argmax_output1, argmax_output2,
        per_class_threshold_output1, per_class_threshold_output2,
        ignore_label,):
    per_class_filter_output1 = argmax_output1.masked_fill(
        mask=max_output1 < per_class_threshold_output1, value=ignore_label)
    per_class_filter_output2 = argmax_output2.masked_fill(
        mask=max_output2 < per_class_threshold_output2, value=ignore_label)
    per_class_filter_output_and = per_class_filter_output1.masked_fill(
        mask=per_class_filter_output1 != per_class_filter_output2, value=ignore_label)
    per_class_filter_output_or = per_class_filter_output_and.flatten().masked_scatter(
        mask=per_class_filter_output1.flatten() == ignore_label,
        source=per_class_filter_output2.flatten()[per_class_filter_output1.flatten() == ignore_label]).masked_scatter(
            mask=per_class_filter_output2.flatten() == ignore_label,
            source=per_class_filter_output1.flatten()[per_class_filter_output2.flatten() == ignore_label]).reshape(
                *per_class_filter_output_and.shape)
    return per_class_filter_output1, per_class_filter_output2, per_class_filter_output_and, per_class_filter_output_or


def fast_hist_torch(a, b, n):
    k = (a >= 0) & (a < n) & (b >= 0) & (b < n)
    return torch.bincount(n * a[k].int() + b[k], minlength=n ** 2).reshape(n, n)


def fast_hist_batch(a, b, n):
    # B x WH
    k = (a >= 0) & (a < n) & (b >= 0) & (b < n)
    mat = n*a+b
    hist = torch.zeros((*mat.shape[:-1], mat.max().item()+1),
                       dtype=mat.dtype,
                       device=mat.device)
    hist = hist.scatter_add(-1, mat, k.to(hist.dtype))
    del mat
    return hist[..., :n**2].reshape(*k.shape[:-1], n, n)


def per_class_iu_torch(hist):
    return hist.diag().float() / (hist.sum(1) + hist.sum(0) - hist.diag()).float()


def per_class_iu_batch(hist):
    return hist.diagonal(0, -2, -1).float() / (hist.sum(-2) + hist.sum(-1) - hist.diagonal(0, -2, -1)).float()


def load_gt(name, gt_dir, set):
    gt_labelIds = torch.stack([
        torch.from_numpy(np.array(Image.open(osp.join(
            gt_dir, set, '_'.join(n.split('_')[:-1]+['gtFine', 'labelIds.png']))
        )))
        for n in name])
    return gt_labelIds


def label_mapping(gt_labelIds, mapping):
    gt_trainIds = gt_labelIds.clone()
    for inx, (labelId, trainId) in enumerate(mapping):
        gt_trainIds.masked_scatter_(
            gt_labelIds == labelId,
            torch.full_like(gt_labelIds, trainId).to(gt_labelIds.device))
    return gt_trainIds


def compute_filtered_output_a_b_and_or(
        max_output_ab, argmax_output_ab,
        ignore_label=255, C=19, N=10, ignore_class=False):
    assert len(max_output_ab) == 2 and len(argmax_output_ab) == 2
    if not isinstance(max_output_ab, torch.Tensor):
        max_output_ab = torch.stack(max_output_ab)
    if not isinstance(argmax_output_ab, torch.Tensor):
        argmax_output_ab = torch.stack(argmax_output_ab)
    assert max_output_ab.shape == argmax_output_ab.shape

    if max_output_ab.dim() == 3:
        _, B, WH = max_output_ab.shape
        BB = B*2
    elif max_output_ab.dim() == 4:
        _, B, W, H = max_output_ab.shape
        BB, WH = B*2, W*H
    else:
        assert False

    # BBx(WH|WxH)
    max_output = torch.cat([*max_output_ab])
    argmax_output = torch.cat([*argmax_output_ab])

    # BBxNx(C|1)
    thr_tb = compute_threshold_table(
        # BBx(WH|WxH)
        max_output,
        argmax_output,
        C=C, N=N, ignore_class=ignore_class,
    )

    # [BBxNx(C|1)].gather(2, BBxNxWH) ) -> BBxNxWH
    # BBxNxWH -> BBxNx(WH|WxH)
    thr_output = thr_tb.gather(
        2,
        # [BBx(WH|WxH) -> BBx1xWH].repeat(1xNx1) -> BBxNxWH
        argmax_output.reshape(BB, 1, WH).repeat(1, N, 1)
    ).reshape(BB, N, *argmax_output.shape[1:])

    # stack(4x(NxBx(WH|WxH)), 0) -> 4xNxBx(WH|WxH)
    filtered_output = torch.stack(
        # 4x(NxBx(WH|WxH))
        per_class_filter_two_output(
            *max_output.split(B),
            *argmax_output.split(B),
            # stack(Nx(BBx1x(WH|WxH))) -> NxBBx1x(WH|WxH)
            # [NxBBx1x(WH|WxH)].squeeze(2).split(B, 1) -> 2x(NxBx(WH|WxH))
            *torch.stack(
                # [BBxNx(WH|WxH)].split(1, 1) -> Nx(BBx1x(WH|WxH))
                thr_output.split(1, 1)).squeeze(2).split(B, 1),
            ignore_label,),
        dim=0)
    # 4xNxBx(WH|WxH)
    return filtered_output


def main():
    B, W, H = 1, 1024, 1024
    num_classes = 19
    gt_labelIds = torch.empty((B*2, W, H)).random_(255).long().cuda()
    output = torch.randn((B*2, 255, W, H)).cuda()
    max_output, argmax_output = output.max(1)

    filtered_output = compute_filtered_output_a_b_and_or(
        max_output.split(B), argmax_output.split(B)).repeat(4, 1, 1, 1, 1)
    import time
    t = time.time()
    hist_ = fast_hist_batch(gt_labelIds.flatten(-2),
                            filtered_output.flatten(-2), 19)
    print(time.time()-t)
    t = time.time()
    IoUs = torch.stack(list(map(
        lambda h: per_class_iu_torch(h),
        hist_.reshape(-1, num_classes, num_classes)
    ))).reshape(16, -1, num_classes)
    print(time.time()-t)
    t = time.time()
    IoUs = per_class_iu_batch(hist_)
    print(time.time()-t)
    t = time.time()
    mIoU = np.nanmean(IoUs.cpu().numpy(), axis=-1)

    del hist_
    torch.cuda.empty_cache()

    t = time.time()
    hist = torch.empty(16*10, B, 19, 19).long()
    for i, o in enumerate(filtered_output.flatten(0, -4)):
        for j in range(B):
            # print(i, j, gt_labelIds[j].shape, o[j].shape)
            hist[i, j, :, :] = fast_hist_torch(gt_labelIds[j], o[j], 19)
    hist = hist.reshape(16, 10, B, 19, 19)
    print(time.time()-t)
    del hist
    torch.cuda.empty_cache()

    t = time.time()
    hist = torch.stack(list(map(
        lambda output: torch.stack(list(map(
            lambda gt, pred: fast_hist_torch(
                gt, pred, 19),
            gt_labelIds, output))),
        filtered_output.flatten(0, -4))))
    print(time.time()-t)
    del hist
    torch.cuda.empty_cache()
    return 0


if __name__ == '__main__':
    main()
