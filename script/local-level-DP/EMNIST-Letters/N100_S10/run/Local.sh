args=(--alg Local
    #  dataset configuration
    --dataset emnist_l
    --num_classes 20
    #  model configuration
    --model mlp
    #  experiment configuration
#     --data_augmentation
    --epochs 1
    --num_users 100
    --shard_per_user 10
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
    --ray_gpu_fraction .25
    #  test configuration
    #  train configuration
#     --verbose
    --lr 1e-2
    --batch_size 100
    --local_ep 500
    --momentum 0
    )

python main.py "${args[@]}"