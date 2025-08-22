OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 train_opt.py \
--config cfgs/density/FDST.yaml \
--device cuda | tee -a training1.log