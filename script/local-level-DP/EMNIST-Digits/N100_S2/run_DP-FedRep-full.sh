args=(--alg DP_FedRep
    #  model configuration
    --model cnn
    #  dataset configuration
    --dataset cifar10
    --shard_per_user 2
    --num_classes 10
    #  experiment configuration
    --data_augmentation
    --data_augmentation_multiplicity 16
    --epochs 20
    --seed 1
    --num_users 100
    #  DP configuration
    #      --disable-dp
    --epsilon 1
    --delta 1e-5
    --dp_clip .2
    #  save/load configuration
    #  backend configuration
#     --gpu 0-1-2-3
    --use_ray
    --ray_gpu_fraction .3
    #  test configuration
    --print_freq 2
    #  train configuration
    --frac_participate 1
    --batch_size 50
    --MAX_PHYSICAL_BATCH_SIZE 5
    --local_ep 5
    # --verbose
    # algorithm specific configuration
    --lr 1e-2
    --lr-head 1e-2
    --local_head_ep 15
    )

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py "${args[@]}"
