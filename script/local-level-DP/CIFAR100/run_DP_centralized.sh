args=(--alg Local
    #  dataset configuration
    --dataset cifar100
    --num_classes 100
    #  model configuration
    --model cnn
    #  experiment configuration
#     --data_augmentation
    --epochs 1
    --num_users 1
    --shard_per_user 100
    --seed 1
    --n_runs 1
    #  DP configuration
    --dp_type local-level-DP
    --disable-dp
    --epsilon 1
    --delta 1e-5
    --dp_clip 1
    #  save/load configuration
    #  backend configuration
    --MAX_PHYSICAL_BATCH_SIZE 500
#     --use_ray
#     --ray_gpu_fraction .3
    #  test configuration
    #  train configuration
#     --verbose
    --lr 1e-2
    --batch_size 1000
    --local_ep 500
    --momentum 0
    )

CUDA_VISIBLE_DEVICES=0 python main.py "${args[@]}"