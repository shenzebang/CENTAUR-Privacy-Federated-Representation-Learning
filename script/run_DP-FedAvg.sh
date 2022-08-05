args=(DP-FedAvg-ft.py
    #  model configuration
     --model cnn
    #  dataset configuration
     --dataset cifar10\
     --shard_per_user 2\
     --num_classes 10\
    #  experiment configuration
#      --data_augmentation
     --epochs 100\
     --seed 1\
     --num_users 100\
    #  DP configuration
#      --disable-dp\
     --epsilon 1\
     --delta 1e-5\
     --dp_clip 1\
    #  save/load configuration
    #  backend configuration
     --gpu 0-1-2-3
     --use_ray
     --ray_gpu_fraction 0.3
    #  test configuration
    #  train configuration
     --lr 1e-1
     --lr-head 1e-2
     --ft-ep 15
     --local_ep 1
     --batch_size 4000
     --MAX_PHYSICAL_BATCH_SIZE 100
     --verbose
     )

python "${args[@]}"
