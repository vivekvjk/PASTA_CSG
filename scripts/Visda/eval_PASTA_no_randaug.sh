RUN_PATH="runs/VisDA-Res101-CSG.stg3.4.w0.1-APool.True-Aug.True-chunk1-mlpTrue.K65536-LR1.00E-04.bone0.1-epoch30-batch32-seed0-PASTA-mode_prop-alpha_10.0-beta_0.5-k_1.0-no_randaug/Nov-21-20_17-47-1669070856/model_best.pth.tar"
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
--use_PASTA \
--PASTA_mode prop \
--PASTA_alpha 10 \
--PASTA_beta 0.5 \
--PASTA_k 1 \
--no_randaug \
--resume $RUN_PATH \
--evaluate