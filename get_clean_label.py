import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image


palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def get_clean_label(model, images, pslabel, interp_target, num_classes, ignore_label, policy, lambda_clean_sample, gpu):
    model.eval()
    image_cols = []

    with torch.no_grad():
        output34 = torch.cat(model(images))[2:]
        # 4xBxCxWxH
        poten34 = interp_target(output34)
        poten34 = poten34.reshape(2, -1, *poten34.shape[1:])
        log_prob34 = F.log_softmax(poten34, dim=-3)
        prob34 = F.softmax(poten34, dim=-3)
        for output, output_name in zip([pslabel]+[*prob34.argmax(dim=2)], ['pslabel', 'pred3', 'pred4']):
            col = to_pil_image(output.cpu().int()).convert('P')
            col.putpalette(palette)
            # col.save('test_noisy/{}_color.png'.format(output_name))
            image_cols.append(('{}_color.png'.format(output_name), col))

        if 'JoCoR' in policy:
            onehot_pslabel = pslabel[:, None] == torch.arange(
                num_classes)[None, :, None, None].to(gpu)
            # KLDivLoss(x,y) = y * [log(y) - log(x)] = KL[y|x]
            kldiv_loss = torch.stack([
                nn.KLDivLoss(reduction='none')(lp1, p2)
                if e1 != e2 else
                nn.KLDivLoss(reduction='none')(
                    lp1, onehot_pslabel.float())
                for e1, lp1 in enumerate(log_prob34)
                for e2, p2 in enumerate(prob34)
            ]).reshape(2, 2, *prob34.shape[1:]).sum(3)
            loss_mat = kldiv_loss * torch.from_numpy(np.array(
                lambda_clean_sample))[:, :, None, None, None].cuda(gpu).float()
            loss34 = loss_mat.sum(1)
            del kldiv_loss, loss_mat
        elif 'CoTeaching' in policy:
            loss_seg_target34 = list(map(
                lambda poten: nn.CrossEntropyLoss(
                    reduction='none', ignore_index=ignore_label)(poten, pslabel),
                # lambda pred_target: loss_calc(
                #     pred_target, pslabel, ignore_label, gpu),
                poten34))

            loss34 = torch.stack(loss_seg_target34)
        else:
            raise NotImplementedError
        # loss34
        for output, output_name in zip(loss34.cpu(), ['loss3', 'loss4']):
            output -= output.min()
            output /= output.max()
            output *= 255
            gray = to_pil_image(output.int()).convert('L')
            # gray.save('test_noisy/{}_gray.png'.format(output_name))
            image_cols.append(('{}_gray.png'.format(output_name), gray))

        if 'plus_cls' in policy:
            # for per class
            loss34 = loss34.unsqueeze(2).repeat(
                1, 1, num_classes, 1, 1)
            mask34 = (pslabel != ignore_label)[None, :, None, :, :] \
                * (prob34.argmax(2, keepdim=True) ==
                    torch.arange(num_classes).to(gpu)[None, None, :, None, None])
        elif 'plus' in policy:
            # ignore class
            mask34 = (pslabel != ignore_label)[
                None, :].repeat(2, 1, 1, 1)
        elif 'cls' in policy:
            # for per class
            loss34 = loss34.unsqueeze(2).sum(0, keepdim=True).repeat(
                2, 1, num_classes, 1, 1)
            mask34 = (pslabel != ignore_label)[None, :, None, :, :] \
                * (prob34.argmax(2, keepdim=True) ==
                    torch.arange(num_classes).to(gpu)[None, None, :, None, None])
        else:
            # ignore class, ignore classifier
            loss34 = loss34.sum(
                0, keepdim=True).repeat(2, 1, 1, 1)
            mask34 = (pslabel != ignore_label)[
                None, :].repeat(2, 1, 1, 1)
        # mask34: ignore False as -inf for sorting, count True for quantile
        for output, m, output_name in zip(prob34.argmax(dim=2).cpu(), mask34.cpu(), ['pred3', 'pred4']):
            if m.dim() == 4:
                m = m.sum(1).bool()
            elif m.dim() != 3:
                assert False
            output.masked_fill_(~m, float(ignore_label))
            col = to_pil_image(output.int()).convert('P')
            col.putpalette(palette)
            # col.save('test_noisy/masked_{}.png'.format(output_name))
            image_cols.append(('masked_{}.png'.format(output_name), col))

        masked_fill_loss34 = loss34.masked_fill(
            ~mask34, float('inf'))
        sorted_loss34 = masked_fill_loss34.flatten(
            -2).sort(-1)[0]
        mask_sum34 = mask34.flatten(
            -2).sum(-1, keepdim=True).float()
        loss34_quantile = sorted_loss34.gather(
            -1, (mask_sum34*0.5).round().long())

        if 'cls' not in policy:
            pred34_filtered = masked_fill_loss34 < loss34_quantile[..., None]
        else:
            loss34_thld_mask = masked_fill_loss34 < loss34_quantile[..., None]
            pred34 = prob34.argmax(2)
            pred34_filtered = pred34.masked_fill(
                ~loss34_thld_mask.gather(
                    2, pred34.unsqueeze(2)).squeeze(2),
                float(ignore_label))
            del loss34_thld_mask, pred34
        # pred34_filtered: small loss (True|class), large loss (False|ignore_label)
        for output, output_name in zip(pred34_filtered.squeeze(2).cpu(), ['pred3', 'pred4']):
            if output.dtype == torch.int64:
                col = to_pil_image(output.int()).convert('P')
                col.putpalette(palette)
            else:
                col = to_pil_image(output.int()*255).convert('L')
            # col.save(
            #     'test_noisy/masked_filtered_{}.png'.format(output_name))
            image_cols.append(
                ('masked_filtered_{}.png'.format(output_name), col))

        if 'plus_cls' in policy:
            pred_filtered_3and4 = pred34_filtered[0].masked_fill(
                pred34_filtered[0] != pred34_filtered[1],
                float(ignore_label)
            )

            pred_filtered_3_minus_3and4 = pred34_filtered[0].masked_fill(
                pred34_filtered[0] == pred_filtered_3and4,
                float(ignore_label)
            )
            pred_filtered_4_minus_3and4 = pred34_filtered[1].masked_fill(
                pred34_filtered[1] == pred_filtered_3and4,
                float(ignore_label)
            )

            selected_sample_for3 = \
                pred_filtered_4_minus_3and4 != float(ignore_label)
            selected_sample_for4 = \
                pred_filtered_3_minus_3and4 != float(ignore_label)

            del pred_filtered_3and4, pred_filtered_3_minus_3and4, pred_filtered_4_minus_3and4
        elif 'plus' in policy:
            pred_filtered_3and4 = pred34_filtered[0].masked_fill(
                pred34_filtered[0] != pred34_filtered[1],
                float(False)
            )

            pred_filtered_3_minus_3and4 = pred34_filtered[0].masked_fill(
                pred34_filtered[0] == pred_filtered_3and4,
                float(False)
            )
            pred_filtered_4_minus_3and4 = pred34_filtered[1].masked_fill(
                pred34_filtered[1] == pred_filtered_3and4,
                float(False)
            )

            selected_sample_for3 = \
                pred_filtered_4_minus_3and4 != float(False)
            selected_sample_for4 = \
                pred_filtered_3_minus_3and4 != float(False)

            del pred_filtered_3and4, pred_filtered_3_minus_3and4, pred_filtered_4_minus_3and4
        elif 'cls' in policy:
            selected_sample_for3, selected_sample_for4 = \
                pred34_filtered != float(ignore_label)
        else:
            selected_sample_for3, selected_sample_for4 = \
                pred34_filtered != float(False)
        selected_samples = [
            selected_sample_for3,
            selected_sample_for4,
        ]
        for output, m, output_name in zip(prob34.argmax(dim=2).cpu(), selected_samples, ['pred3', 'pred4']):
            output.masked_fill_(~m.cpu(), float(ignore_label))
            col = to_pil_image(output.int()).convert('P')
            col.putpalette(palette)
            # col.save('test_noisy/selected_{}.png'.format(output_name))
            image_cols.append(('selected_{}.png'.format(output_name), col))

        # del output34, poten34, log_prob34, prob34
        # del loss34, mask34
        # del masked_fill_loss34, sorted_loss34, mask_sum34, loss34_quantile
        # del pred34_filtered
    # print('get clean-label', torch.cuda.memory_summary())
    return selected_samples, image_cols
