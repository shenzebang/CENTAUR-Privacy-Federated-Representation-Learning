args=(--alg Local
    #  dataset configuration
    --dataset emnist_d
    --num_classes 10
    #  model configuration
    --model mlp
    #  experiment configuration
#     --data_augmentation
    --epochs 1
    --num_users 2000
    --shard_per_user 2
    --seed 1
    --n_runs 1
    #  DP configuration
    --disable-dp
    #  save/load configuration
    #  backend configuration
#     --gpu 0-1-2-3
    --use_ray
    --ray_gpu_fraction .25
    #  test configuration
    #  train configuration
#     --verbose
    --lr 1e-2
    --batch_size 100
    --local_ep 200
    --momentum 0
    )

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py "${args[@]}"