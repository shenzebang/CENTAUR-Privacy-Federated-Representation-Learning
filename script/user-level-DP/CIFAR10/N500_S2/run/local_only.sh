args=(--alg Local
    #  dataset configuration
    --dataset cifar10
    --num_classes 10
    #  model configuration
    --model cnn
    #  experiment configuration
#     --data_augmentation
    --epochs 1
    --num_users 500
    --shard_per_user 2
    --seed 1
    --n_runs 1
    #  DP configuration
    --disable-dp
    --epsilon 1
    --delta 1e-5
    --dp_clip 1
    #  save/load configuration
    #  backend configuration
    --use_ray
    --ray_gpu_fraction .25
    #  test configuration
    #  train configuration
#     --verbose
    --lr 1e-2
    --batch_size 64
    --local_ep 500
    --momentum 0
    )

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py "${args[@]}"