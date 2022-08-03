args=(main_fedrep.py
    # algorithm configuration
     --alg fedavg\
    #  model configuration
     --model cnn
    #  dataset configuration
     --dataset cifar10\
     --shard_per_user 2\
     --num_classes 10\
    #  experiment configuration
     --data_augmentation
     --arc fl\
     --epochs 100\
     --seed 1\
     --num_users 100\
    #  DP configuration
     --epsilon 1\
     --delta 1e-5\
     --dp_clip 1\
    #  save/load configuration
    #  backend configuration
     --gpu 2\
    #  test configuration
     --test_freq 10
    #  train configuration
     --lr 1e-2
     --local_ep 1
     )

python "${args[@]}"
