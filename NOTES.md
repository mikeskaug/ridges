## NOTES

    rsync -rve "ssh -i ~/.ssh/ridges-DL-spot.pem" ./src ubuntu@ec2-52-90-166-34.compute-1.amazonaws.com:~/ridges/

    conda info --envs

    screen
    source src/set_env.sh
    screen -r


## 2019-04-10

- try higher gamma in focal loss
- don't use generator for validation data, so that we can save distributions/images in TensorBoard
- make sure augmentation works same on images and masks