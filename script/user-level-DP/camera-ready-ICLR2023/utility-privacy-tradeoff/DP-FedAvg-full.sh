for eps in .25 .5 2 4
do
args=(--alg DP_FedAvg_ft
    #  model configuration
    --model cnn
    #  dataset configuration
    --dataset cifar10
    --shard_per_user 5
    --num_classes 10
    #  experiment configuration
#     --data_augmentation
#     --data_augmentation_multiplicity 16
    --epochs 200
    --seed 1
    --num_users 500
    --n_runs 3
    #  DP configuration
    #      --disable-dp
    --dp_type user-level-DP
    --epsilon $eps
    --delta 1e-5
    --dp_clip .01
    #  save/load configuration
    #  backend configuration
    --use_ray
    --ray_gpu_fraction 0.33
    #  test configuration
    #  train configuration
    --frac_participate 1
    --batch_size 100
    --local_ep 1
    # --verbose
    --print_freq 20
    # algorithm specific configuration
    --lr 1e-2
    --lr-head 1e-2
    --local_head_ep 15
    --description ICLR2023_CR_Utility_Privacy_Tradeoff
    )

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py "${args[@]}"
done