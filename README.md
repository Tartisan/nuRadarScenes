# nuRadarScenes

### Run
```
## Check model in ./models 
## E.g. pointnet_ssg
python train_semseg.py --model=pointnet_sem_seg --epoch=100 --log_dir pointnet_sem_seg
```
### inference
```
python inference.py --model=pointnet_sem_seg
```