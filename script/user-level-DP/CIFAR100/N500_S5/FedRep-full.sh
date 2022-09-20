args=(--alg DP_FedRep
    #  model configuration
    --model cnn
    #  dataset configuration
    --dataset cifar100
    --shard_per_user 5
    --num_classes 100
    #  experiment configuration
    #         --data_augmentation
    #         --data_augmentation_multiplicity 16
    --epochs 400
    --seed 1
    --num_users 500
    #  DP configuration
    --disable-dp
    #  save/load configuration
    #  backend configuration
#     --gpu 0-1-2-3
    --use_ray
    --ray_gpu_fraction .3
    #  test configuration
    #  train configuration
    --frac_participate 1
    --batch_size 10
    --local_ep 1
    # --verbose
    # algorithm specific configuration
    --lr 1e-2
    --lr-head 1e-2
    --local_head_ep 15
    )

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py "${args[@]}"
