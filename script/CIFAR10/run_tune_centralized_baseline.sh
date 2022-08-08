args=(tune_baseline_centralized.py
    #  dataset configuration
    --dataset cifar10
    --num_classes 10
    #  model configuration
    --model cnn
    #  experiment configuration
#     --data_augmentation
    --epochs 600
    --seed 1
    --n_runs 1
    #  DP configuration
#     --disable-dp
    --epsilon 1
    --delta 1e-5
    --dp_clip 1.2
    #  save/load configuration
    #  backend configuration
    --gpu 0
    #  test configuration
    --weight-decay 0
    --momentum 0
    #  train configuration
    --lr .3
    --batch_size 4000
    --MAX_PHYSICAL_BATCH_SIZE 200
    --verbose
    )

python "${args[@]}"