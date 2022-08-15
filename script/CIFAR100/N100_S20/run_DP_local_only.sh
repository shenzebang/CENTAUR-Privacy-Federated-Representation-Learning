args=(--alg Local
    #  dataset configuration
    --dataset cifar100
    --num_classes 100
    #  model configuration
    --model cnn
    #  experiment configuration
#     --data_augmentation
    --epochs 1
    --num_users 100
    --shard_per_user 20
    --seed 1
    --n_runs 1
    #  DP configuration
    --disable-dp
    --epsilon 1
    --delta 1e-5
    --dp_clip 1
    #  save/load configuration
    #  backend configuration
#     --gpu 0-1-2-3
    --use_ray
    --ray_gpu_fraction .3
    #  test configuration
    #  train configuration
#     --verbose
    --lr 1e-1
    --batch_size 4000
    --local_ep 500
    --momentum 0
    )

python main.py "${args[@]}"