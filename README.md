# Partial-DANN

Basic usage
```powershell
python .\segmentation.py --root 'C:\Users\User\Desktop\TSM_Project\data\amos22' --output checkpoints --modality ct --batch_size 1 --lr 0.0001 --optim AdamW --max_iter 10000  --eval_step 500 --cache_rate 0.5 --num_workers 2 
```