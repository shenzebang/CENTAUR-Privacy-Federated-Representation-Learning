args=(--alg DP_FedRep
    #  model configuration
    --model cnn
    #  dataset configuration
    --dataset cifar100
    --shard_per_user 20
    --num_classes 100
    #  experiment configuration
    #      --data_augmentation
    --epochs 400
    --seed 1
    --num_users 100
    #  DP configuration
    #      --disable-dp
    --epsilon 1
    --delta 1e-5
    --dp_clip 1
    #  save/load configuration
    #  backend configuration
#     --use_ray
#     --ray_gpu_fraction .5
    #  test configuration
    #  train configuration
    --frac_participate 1
    --batch_size 4000
    --MAX_PHYSICAL_BATCH_SIZE 64
    --local_ep 1
    # --verbose
    # algorithm specific configuration
    --lr-rep 1e-1
    --lr-head 1e-2
    --local_head_ep 15
    )

CUDA_VISIBLE_DEVICES=0 python main.py "${args[@]}"
