Pytorch deepsleepnet 
tensorflow version https://github.com/akaraspt/tinysleepnet
Todo
- [x] gradient clipping
- [ ] data augmentation
- [ ] class weight adjustment

fold 0 test accuracy = 0.88
fold 1 test accuracy = 0.8396

config = {
    n_fold 20
    fold_idx 
    n_epochs 200
}

the stride of the first filter needs to be sampling rate//16 <strong>not sampling rate//4</strong>
https://github.com/akaraspt/tinysleepnet/issues/2
