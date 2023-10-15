# Partial-DANN

Basic usage
```powershell
python main.py 
--root 'C:\Users\User\Desktop\TSM_Project\data\amos22' 
--output checkpoints 
--modality ct 
--batch_size 1 
--lr 0.0001 
--optim AdamW 
--max_iter 10000  
--eval_step 500 
--cache_rate 0.5 
--num_workers 2 
```

#### Fully-labeled dataset
* **Segmentation**
```
python main.py --module segmentation --modality MODALITY --loss dice2
``` 

* **Segmentation with domain adaptation**
```
python main.py --module dann --modality ct+mr --loss dice2
```

#### Partially-labelled dataset
* **Segmentation**
```
python main.py --module segmentation --modality MODALITY --masked --loss LOSS*
``` 
*Loss cannot be `tal` if the modality is `ct+mr`

* **Segmentation with domain adaptation**

One can choose to mask the loaded image (1) before training
```
python main.py --module dann --modality ct+mr --masked --loss LOSS
```
or (2) when calculating loss and during validation
```
python main.py --module dann --modality ct+mr --loss tal
```

|   Label masked  |   Target Adaptative Loss  |       Masked Data       |
|:---------------:|:-------------------------:|:-----------------------:|
|  Training data  |             X             |            V            |
| Prediction/Loss |             V             |            X            |
| Validation data | V <br>(by postprocessing) | V <br>(by mask_mapping) |
|   Testing data  |             X             |            X            |