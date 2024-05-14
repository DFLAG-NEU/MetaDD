# MetaDD
Implementation of our [Domain Diversity Based Meta Learning for Continual Person Re-identification]

## Dataset prepration
Please follow [Torchreid_Dataset_Doc](https://kaiyangzhou.github.io/deep-person-reid/datasets.html) to download datasets and unzip them to your data path (we refer to 'machine_dataset_path' in train_test.py). Alternatively, you could download some of never-seen domain datasets in [DualNorm](https://github.com/BJTUJia/person_reID_DualNorm).

## Train 
python train_test.py

## Test
After training, you can perform testing by 'python train_test.py --mode test' based on your saved model.
Our [trained model](https://drive.google.com/file/d/1TTsqFGi7ghJdMaudvo9E-1WgTHN5SOEr/view?usp=sharing) is also provided.

# Acknowledgement
Our work is based on the code of Nan Pu et al. ([LifeLongReID](https://github.com/TPCD/LifelongReID)) and our previous work ([CKP](https://github.com/DFLAG-NEU/ContinualReID)). Many thanks to Nan Pu et al. for building the foundations for continual person re-identification.
