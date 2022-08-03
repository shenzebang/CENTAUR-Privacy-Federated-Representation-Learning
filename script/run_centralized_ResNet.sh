args=(baseline_centralized.py
    #  dataset configuration
    --dataset cifar10
    --num_classes 10
    #  model configuration
    --model resnet
    #  experiment configuration
#     --data_augmentation
    --epochs 20
    --seed 1
    --n_runs 1
    #  DP configuration
#     --disable-dp
    --epsilon 50
    --delta 1e-5
    --dp_clip 1.2
    #  save/load configuration
    #  backend configuration
    --gpu 3
    #  test configuration
    #  train configuration
    --lr 1e-1
    --batch_size 100
    )

python "${args[@]}"