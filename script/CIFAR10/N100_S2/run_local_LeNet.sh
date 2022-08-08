args=(baseline_local.py
    #  dataset configuration
    --dataset cifar10
    --num_classes 10
    #  model configuration
    --model cnn
    #  experiment configuration
    --data_augmentation
    --epochs 1
    --num_users 20
    --shard_per_user 5
    --seed 1
    --n_runs 1
    #  DP configuration
    --disable-dp
    --epsilon 1
    --delta 1e-5
    --dp_clip 1.2
    #  save/load configuration
    #  backend configuration
    --gpu 0
    #  test configuration
    #  train configuration
#     --verbose
    --lr 1e-3
    --batch_size 50
    --local_epochs 500
    --momentum 0.9
    )

python "${args[@]}"