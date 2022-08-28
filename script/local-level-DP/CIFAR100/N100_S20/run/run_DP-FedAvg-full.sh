args=(--alg DP_FedAvg_ft
    #  model configuration
    --model cnn
    #  dataset configuration
    --dataset cifar100
    --shard_per_user 20
    --num_classes 100
    #  experiment configuration
    #      --data_augmentation
    --epochs 200
    --seed 1
    --num_users 100
    #  DP configuration
    #      --disable-dp
    --epsilon 1
    --delta 1e-5
    --dp_clip .5
    #  save/load configuration
    #  backend configuration
#     --gpu 0-1-2-3
    --use_ray
    --ray_gpu_fraction .33
    #  test configuration
    #  train configuration
    --frac_participate 1
    --batch_size 50
    --MAX_PHYSICAL_BATCH_SIZE 25
    --local_ep 1
    # --verbose
    # algorithm specific configuration
    --lr 1e-2
    --lr-head 1e-2
    --ft-ep 15
    )

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py "${args[@]}"
