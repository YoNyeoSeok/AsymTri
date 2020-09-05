# run-20200817_051939-9ab2hb8f norm2 orth[]
# run-20200817_052030-2rqsiz4u norm2 orth[1]
# run-20200818_083558-380q72nj norm2 orth[1,2,3]
# run-20200831_003751-fyqmpvrz norm2 orth[2,3]

python train_gta2cityscapes_dualco.py --model DeepLabDualCo --lambda_loss .0 .0 .5 .5 \
    --restore-from wandb/run-20200817_051939-9ab2hb8f/GTA5_70000.pth \
    --pslabel-dir-target psLabel/cityscapes/9ab2hb8f/GTA5_70000/class_90_100_and \
    --use-wandb
python train_gta2cityscapes_dualco.py --model DeepLabDualCo --lambda_loss .25 .25 .25 .25 \
    --restore-from wandb/run-20200817_051939-9ab2hb8f/GTA5_70000.pth \
    --pslabel-dir-target psLabel/cityscapes/9ab2hb8f/GTA5_70000/class_90_100_and \
    --use-wandb

python train_gta2cityscapes_dualco.py --model DeepLabDualCo --lambda_loss .0 .0 .5 .5 \
    --restore-from wandb/run-20200817_052030-2rqsiz4u/GTA5_70000.pth \
    --pslabel-dir-target psLabel/cityscapes/2rqsiz4u/GTA5_70000/class_90_100_and \
    --use-wandb --gpu 1
python train_gta2cityscapes_dualco.py --model DeepLabDualCo --lambda_loss .25 .25 .25 .25 \
    --restore-from wandb/run-20200817_052030-2rqsiz4u/GTA5_70000.pth \
    --pslabel-dir-target psLabel/cityscapes/2rqsiz4u/GTA5_70000/class_90_100_and \
    --use-wandb --gpu 1

python train_gta2cityscapes_dualco.py --model DeepLabDualCo --lambda_loss .0 .0 .5 .5 \
    --restore-from wandb/run-20200818_083558-380q72nj/GTA5_70000.pth \
    --pslabel-dir-target psLabel/cityscapes/380q72nj/GTA5_70000/class_90_100_and \
    --use-wandb --gpu 2
python train_gta2cityscapes_dualco.py --model DeepLabDualCo --lambda_loss .25 .25 .25 .25 \
    --restore-from wandb/run-20200818_083558-380q72nj/GTA5_70000.pth \
    --pslabel-dir-target psLabel/cityscapes/380q72nj/GTA5_70000/class_90_100_and \
    --use-wandb --gpu 2

python train_gta2cityscapes_dualco.py --model DeepLabDualCo --lambda_loss .0 .0 .5 .5 \
    --restore-from wandb/run-20200831_003751-fyqmpvrz/GTA5_70000.pth \
    --pslabel-dir-target psLabel/cityscapes/fyqmpvrz/GTA5_70000/class_90_100_and \
    --use-wandb --gpu 3
python train_gta2cityscapes_dualco.py --model DeepLabDualCo --lambda_loss .25 .25 .25 .25 \
    --restore-from wandb/run-20200831_003751-fyqmpvrz/GTA5_70000.pth \
    --pslabel-dir-target psLabel/cityscapes/fyqmpvrz/GTA5_70000/class_90_100_and \
    --use-wandb --gpu 3
