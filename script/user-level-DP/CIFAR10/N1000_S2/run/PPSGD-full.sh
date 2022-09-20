args=(main.py
        #  algorithm configuration
        --alg PPSGD
        #  model configuration
        --model cnn
        #  dataset configuration
        --dataset cifar10
        --shard_per_user 2
        --num_classes 10
        #  experiment configuration
        #      --data_augmentation
        --epochs 400
        --seed 1
        --num_users 500
        #  DP configuration
        #      --disable-dp
        --dp_type user-level-DP
        --epsilon 1
        --delta 1e-5
        --dp_clip 1
        #  save/load configuration
        #  backend configuration
        --use_ray
        --ray_gpu_fraction 0.3
        #  test configuration
        #  train configuration
        --batch_size 4000
        --MAX_PHYSICAL_BATCH_SIZE 100
        --local_ep 1
        # --verbose
        # algorithm specific configuration
        --lr 1e-3
        --lr_l 1e-1
     )

CUDA_VISIBLE_DEVICES=0,1,2,3 python "${args[@]}"
