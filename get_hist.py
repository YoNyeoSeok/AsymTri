import torch
import torch.nn.functional as F

from eval_iou import compute_filtered_output_a_b_and_or, fast_hist_torch


def get_hist(model, images, gt_trainIds, interp_target_gt, num_classes, ignore_label, gpu):
    model.cuda(gpu)
    with torch.no_grad():
        # 4BxCxWxH
        output1234 = torch.cat(model(images))
        # 4xBxCxWxH
        poten1234 = interp_target_gt(output1234)
        poten1234 = poten1234.reshape(4, -1, *poten1234.shape[1:])
        prob1234 = F.softmax(poten1234, dim=-3)
        # 4xBxWxH
        confid1234, pred1234 = prob1234.max(2)

        # 16xNxBxWxH
        filtered_pred_1_2_1and2_1or2_3_4_3and4_3or4 = torch.cat(list(map(
            lambda max_output_ab, argmax_output_ab:
                # 8xNxBxWxH
                torch.cat(list(map(
                    # 4xNxBxWxH
                    lambda igc: compute_filtered_output_a_b_and_or(
                        max_output_ab, argmax_output_ab,
                        ignore_label=ignore_label, C=num_classes, N=10, ignore_class=igc),
                    [True, False]
                ))),
            confid1234.split(2), pred1234.split(2)
        )))

        # WHY BATCH IS MORE SLOWER?
        # [16xNxBx19x19].sum(2) -> 16xNx19x19
        hist = torch.stack(list(map(
            lambda pred_batch: torch.stack(list(map(
                lambda gt, pred_batch: fast_hist_torch(
                    gt, pred_batch, num_classes),
                gt_trainIds, pred_batch
            ))),
            filtered_pred_1_2_1and2_1or2_3_4_3and4_3or4.flatten(0, -4)
        ))).reshape(*filtered_pred_1_2_1and2_1or2_3_4_3and4_3or4.shape[:-2], num_classes, num_classes).sum(2)
        # WHY BATCH IS MORE SLOWER?
        # [16xNxBx19x19].sum(2) -> 16xNx19x19
        # hist = fast_hist_batch(
        #     # [BxWxH].flatten(-2) -> BxWH
        #     gt_trainIds.flatten(-2),
        #     # [16xNxBxWxH].flatten(-2) -> 16xNxBxWH
        #     filtered_pred_1_2_1and2_1or2_3_4_3and4_3or4.flatten(
        #         -2),
        #     num_classes).sum(2)

    return hist
