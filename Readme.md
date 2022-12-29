# AntiVP-Pruning

## Fine-tune model
    python main.py --model mobilebert --dataset sst-2 --epoch 10 --log_prefix finetune

## Fine-tune model while freeze feature extractors
    python main.py --model mobilebert --dataset sst-2 --epoch 10 --freeze_bert --log_prefix finetune

## Evaluate model
    python main.py --model mobilebert --dataset sst-2 --eval_only --load_cp_path /mnt/sting/yewon/antivp-pruning/log/sst-2/mobilebert/1024_finetune_mobilebert/cp/cp_e10.pth.tar 

## Evaluate model with pruning
    python main.py --model mobilebert --dataset sst-2 --eval_only --load_cp_path /mnt/sting/yewon/antivp-pruning/log/sst-2/mobilebert/1024_finetune_mobilebert/cp/cp_e10.pth.tar --prune --seed 1

> Check pruning parameters in main.py to revise hyperparameters
