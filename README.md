# Vesuvius Challenge - Ink Detection (9th place  solution)
https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection

For solution discussion, refer to https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417361


## 1. Hardware  
- GPU: 2x Nvidia Quadro RTX 8000, each with VRAM 48 GB
- CPU: Intel® Xeon(R) Gold 6240 CPU @ 2.60GHz, 72 cores
- Memory: 376 GB RAM

## 2. OS 
- ubuntu 18.04.5 LTS

## 3. Set Up Environment
- Install Python >=3.8.10
- Install requirements.txt in the python environment
- Set up the directory structure as shown below.
``` 
└── solution
    ├── src 
    ├── results
    ├── data
    |   ├── vesuvius-challenge-ink-detection
    |   |   ├── test
    │   |   ├── train
    │   |   ├── sample_submission.csv
    │   |   
    |   ├── pretrained   
    |       ├── pvt_v2_b3.pth
    | 
    ├── r050_resnet34-unet-mean32-pool-05.sh   
    ├── r090-pvtv2-daformer-pool-02a.sh  
    ├── hyper-parameters.pdf
    ├── LICENSE 
    ├── README.md 
```

- The dataset "vesuvius-challenge-ink-detection" can be downloaded from Kaggle:  
https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/data

- Pretrained model can be download from [PVT (Pyramid Vision Transformer)](https://github.com/whai362/PVT) repository:  
https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth


## 4. Training the model

### Warning !!! training output will be overwritten to the "solution/results" folder 

- Use the 2 commands to train the weights of the deep neural nets. The bash script will call "run_train.py" with appropriate configure file.
```
usage: python run_train.py <configure>

#pwd = /solution
>> bash ./r050_resnet34-unet-mean32-pool-05.sh
>> bash ./r090-pvtv2-daformer-pool-02a.sh
```
- This will produce the 4 model files used in submission:
``` 
└── solution
    ├── ...
    ├── results
        ├── r050_resnet34-unet-mean32-pool-05
        |   ├── fold-1 
        |   |   ├── stage2_0/checkpoint/00018924.model.pth
        |   |      
        |   ├── fold-2aa
        |       ├── stage2_0/checkpoint/00014850.model.pth
        |   
        ├── r090-pvtv2-daformer-pool-02a
            ├── fold-1 
            |   ├── stage2_0/checkpoint/00029376.model.pth
            |      
            ├── fold-2aa
                ├── stage2_0/checkpoint/00009159.model.pth
      
 
```
- optional: if you are interested in the parameters used in training the different stages of the model , you can refer to "solution/hyper-parameters.pdf"


## 5. Submission notebook
 - The public submission notebook is at:
https://www.kaggle.com/code/hengck23/9th-place-final-ensemble?scriptVersionId=133572613

   [private / public score] 0.654354 / 0.738625


## 6. Local validation

- Run the "run_infer_ms.py" python sript for local validation.

```commandline
usage: python run_infer_ms.py <configure> <model_file>

#pwd = /solution
>> python src/r050_resnet34-unet-mean32-pool-05/run_infer_ms.py config_fold2aa_stage2_0 results/r050_resnet34-unet-mean32-pool-05/fold-2aa/stage2_0/checkpoint/fold-2aa-Resnet34MeanPool-00014850.model.pth

```

## 7. Reference train and validation results
- Reference results can be found at the google share drive. It includes the weight files, train/validation logs and visualisation images. You can use this to check your results.

  [[google drive link (1.8 GB)](https://drive.google.com/drive/folders/1LF77aNJhFXQDzn_pVclAc9H2z5ZlGWXt?usp=sharing)]

## Authors

- https://www.kaggle.com/hengck23

## License

- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 



