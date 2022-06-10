# OpenPifPaf: Performance vs Efficiency<br />
In this work, we experiment how different backbones would affect the performance of the state-of-the-art bottom-up pose estimation model, and the pros and
cons between the models. We came to conclusion that it makes sense that ShuffleNetV2 Ma et al. (2018) as the original authors choice since it indeed have the
best performance, but we still find valuable insights with the cross-backbone com-
parison. EfficientNetV2 Tan and Le (2021), in our opinion, is the more balanced
choice which maintains comparable performance with ShuffleNetV2 and keeps the
run-time lower than that. We also proposed future work with our finding of this
project.

## Training

## Evaluation

Training: 
To train the Openpifpaf model, you mgiht use the follwoing command:<br />
python train.py<br />
Options:
Learning rate: --lr 
Learning rate decay: --lr-decay


To evaluate your model performance, we will use `eval.py` file to execute.  
Example command: `python3 eval.py `  

```=linux
eval.py:  
	--batch-size BATCH_SIZE   
	processing batch size (default: 1)  
	--checkpoint CHECKPOINT  
	Path to a local checkpoint.
```
	
Plots:<br />
![alt text](all-images/effnet/0008.jpeg) <br />

