RUN_PATH="runs/VisDA-Res101-CSG.stg3.4.w0.1-APool.True-Aug.True-chunk1-mlpTrue.K65536-LR1.00E-04.bone0.1-epoch30-batch32-seed0/Nov-20-20_22-57-1669003047/model_best.pth.tar"
python train.py \
--epochs 30 \
--batch-size 32 \
--lr 1e-4 \
--rand_seed 0 \
--csg 0.1 \
--apool \
--augment \
--csg-stages 3.4 \
--factor 0.1 \
--resume $RUN_PATH \
--evaluate