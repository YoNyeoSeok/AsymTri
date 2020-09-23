import torch
import torch.nn.functional as F

from eval_iou import compute_filtered_output_a_b_and_or


def get_pseudo_label(prev_model, images, interp_target, num_classes, ignore_label, policy, thr):
    # get pseudo-label
    prev_model.eval()
    with torch.no_grad():
        # 4BxCxWxH
        pm_output1234 = torch.cat(prev_model(images))
        # 4BxCxWxH
        pm_poten1234 = interp_target(pm_output1234)
        # 4xBxCxWxH
        pm_poten1234 = pm_poten1234.reshape(
            4, -1, *pm_poten1234.shape[1:])
        pm_prob1234 = F.softmax(pm_poten1234, -3)
        # 4xBxWxH
        pm_confid1234, pm_pred1234 = pm_prob1234.max(2)

        # 16xNxBxWxH
        filtered_pm_pred_1_2_1and2_1or2_3_4_3and4_3or4 = torch.cat(list(map(
            lambda max_output_ab, argmax_output_ab:
                # 8xNxBxWxH
                torch.cat(list(map(
                    # 4xNxBxWxH
                    lambda igc: compute_filtered_output_a_b_and_or(
                        max_output_ab, argmax_output_ab,
                        ignore_label=ignore_label, C=num_classes, N=10, ignore_class=igc),
                    [True, False]
                ))),
            pm_confid1234.split(2), pm_pred1234.split(2)
        )))
        # pslabel = pm_pred (prev model filtered pred)
        pslabel = filtered_pm_pred_1_2_1and2_1or2_3_4_3and4_3or4[
            policy, thr, ]
        #     policy_index.index(args.pslabel_policy),
        #     thr_columns.index(threshold),]
        # del pm_poten1234, pm_prob1234, pm_confid1234, pm_pred1234
        # del filtered_pm_pred_1_2_1and2_1or2_3_4_3and4_3or4
    # print('get pseudo-label', torch.cuda.memory_summary())
    return pslabel
