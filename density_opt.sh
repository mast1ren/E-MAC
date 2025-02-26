OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 train_opt.py \
--config cfgs/density/FDST.yaml \
--device cuda 2>&1 | tee -a training.log