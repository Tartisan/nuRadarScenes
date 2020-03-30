# nuRadarScenes

### Run
```
## Check model in ./models 
## E.g. pointnet2_ssg
python train_semseg.py --model=pointnet2_sem_seg --epoch=100 --log_dir pointnet2_sem_seg
python test_semseg.py --model=pointnet2_sem_seg --epoch=100 --log_dir pointnet2_sem_seg
```
### inference
```
python inference.py --model=pointnet2_sem_seg
```