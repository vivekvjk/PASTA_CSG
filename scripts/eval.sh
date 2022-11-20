RUN_PATH="runs/"
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