import os.path as osp
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import pandas as pd


def output_to_threshold(max_output, N=10):
    device = max_output.device

    reshaped_max_output = max_output.reshape(len(max_output), -1)
    sorted_max_output, _ = reshaped_max_output.sort()

    inf_pad_max_output = torch.cat(
        [sorted_max_output,
         torch.full((len(max_output), 1), float('inf'), device=device)],
        dim=1)
    threshold_idx_output = torch.linspace(
        0, sorted_max_output.shape[1], N+1, device=device)[:-1].round().long()
    threshold_output = inf_pad_max_output[:, threshold_idx_output]
    return threshold_output


def output_to_class_threshold(max_output, argmax_output, N=10, num_classes=19):
    device = max_output.device
    reshaped_max_output, reshaped_argmax_output = list(map(
        lambda output: output.reshape(len(max_output), -1),
        [max_output, argmax_output]
    ))

    sorted_max_output, sorted_idx_max_output = reshaped_max_output.sort()
    sorted_idx_argmax_output = reshaped_argmax_output.gather(
        1, sorted_idx_max_output)
    inf_pad_max_output = torch.cat(
        [sorted_max_output,
            torch.full((len(max_output), 1), float('inf'), device=device)],
        dim=1)

    per_class_threshold_output = torch.empty(
        (len(max_output), num_classes, N), device=device)
    for c in range(num_classes):
        mask = sorted_idx_argmax_output == c
        last_pad_mask = torch.cat(
            [mask, torch.ones((len(max_output), 1), device=device).bool()],
            dim=1)
        threshold_idx_output = torch.cat([
            last_pad_mask[b].nonzero()[
                torch.linspace(0, mask_sum, N+1, device=device)[:-1].round().long()]
            for b, mask_sum in enumerate(mask.sum(1))],
            dim=1).transpose(0, 1)
        threshold_output = inf_pad_max_output.gather(1, threshold_idx_output)

        per_class_threshold_output[:, c] = threshold_output
    return per_class_threshold_output


def filter_two_output(
        max_output1, max_output2,
        argmax_output1, argmax_output2,
        threshold_output1, threshold_output2,
        ignore_label,):
    filter_output1 = argmax_output1.masked_fill(
        mask=max_output1 < threshold_output1, value=ignore_label)
    filter_output2 = argmax_output2.masked_fill(
        mask=max_output2 < threshold_output2, value=ignore_label)
    filter_output_and = filter_output1.masked_fill(
        mask=filter_output1 != filter_output2, value=ignore_label)
    filter_output_or = filter_output_and.flatten().masked_scatter(
        mask=filter_output1.flatten() == 255,
        source=filter_output2.flatten()[filter_output1.flatten() == 255]).masked_scatter(
            mask=filter_output2.flatten() == 255,
            source=filter_output1.flatten()[filter_output2.flatten() == 255]).reshape(
                *filter_output_and.shape)
    return filter_output1, filter_output2, filter_output_and, filter_output_or


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


def per_class_iu_torch(hist):
    return hist.diag().float() / (hist.sum(1) + hist.sum(0) - hist.diag()).float()


def eval_iou(info, model, testgttargetloader, testtargetloader, interp_target_gt, args, index, columns, print_index=-1, print_columns=4):
    num_classes = np.int(info['classes'])
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = torch.from_numpy(
        np.array(info['label2train'])).byte().cuda(args.gpu)
    hist = torch.zeros(
        (len(index), len(columns), num_classes, num_classes)).cuda(args.gpu)

    model.eval()
    with torch.no_grad():
        for ind, (gt_batch, batch) in enumerate(zip(testgttargetloader, testtargetloader)):
            _, _, gt_name = gt_batch
            images, _, name = batch
            assert len(name) == len(gt_name)
            batch_size = len(name)
            for name_, gt_name_ in zip(name, gt_name):
                assert ('_'.join(name_.split('_')[:-1]) ==
                        '_'.join(gt_name_.split('_')[:-2]))

            gt_labelIds = torch.stack(
                [torch.from_numpy(np.array(Image.open(
                    osp.join(args.gt_dir_target, 'val', gt_name_)))).cuda(args.gpu)
                 for gt_name_ in gt_name])
            gt_trainIds = gt_labelIds.clone()

            for inx in range(len(mapping)):
                gt_trainIds.masked_scatter_(
                    gt_labelIds == mapping[inx][0],
                    torch.full_like(
                        gt_labelIds, mapping[inx][1], device=args.gpu)
                )

            images = images.cuda(args.gpu)
            output12 = model(images)
            output12 = list(map(interp_target_gt, output12))

            # save max_output, argmax
            max_output12, argmax_output12 = list(zip(*list(map(
                lambda output: output.max(1), output12))))
            # compute threshold (N=10)
            threshold_output12 = list(
                map(output_to_threshold, max_output12))
            per_class_threshold_output12 = list(
                map(output_to_class_threshold, max_output12, argmax_output12))

            for idx, _ in enumerate(index):
                filter_output_1_2_and_or = filter_two_output(
                    *max_output12,
                    *argmax_output12,
                    *list(map(lambda thr_output: thr_output[:, idx, None, None], threshold_output12)),
                    args.ignore_label,
                )
                per_class_filter_output_1_2_and_or = per_class_filter_two_output(
                    *max_output12,
                    *argmax_output12,
                    *list(map(
                        lambda thr_output, argmax_output:
                        thr_output[:, :, idx].gather(
                            1, argmax_output.reshape(len(argmax_output), -1)
                        ).reshape(*argmax_output.shape),
                        per_class_threshold_output12, argmax_output12)),
                    args.ignore_label,
                )

                # compute iou
                for idx_, pred in enumerate(
                        list(filter_output_1_2_and_or)+list(per_class_filter_output_1_2_and_or)):
                    hist[idx][idx_] += fast_hist_torch(gt_trainIds.flatten(),
                                                       pred.flatten(), num_classes)
            print('{:d} / {:d} ({}, {}): {:0.2f}'.format(
                ind, len(testgttargetloader),
                index[print_index], columns[print_columns],
                100*np.nanmean(per_class_iu_torch(hist[print_index][print_columns]).cpu().numpy())))

        IoUs = np.zeros((len(index), len(columns), num_classes+1))
        for idx, _ in enumerate(index):
            for idx_, _ in enumerate(columns):
                IoUs[idx, idx_, :-1] = per_class_iu_torch(
                    hist[idx][idx_]).cpu().numpy()
        IoUs[:, :, -1] = np.nanmean(IoUs[:, :, :-1], axis=-1)

        dfs = {}
        for ind_class in range(num_classes+1):
            name_class = name_classes[ind_class] if ind_class < num_classes else 'mIoU'
            df = pd.DataFrame(IoUs[:, :, ind_class],
                              index=index, columns=columns)
            dfs.update({name_class: df})

            print('===> {} ({}, {}):\t {:.2f}'.format(
                name_class, index[print_index], columns[print_columns],
                100*IoUs[print_index, print_columns, ind_class]))
    return dfs
