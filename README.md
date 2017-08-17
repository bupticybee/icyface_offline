# icyface_offline

This project contains the offline part of the icyface platform, including data preprocessing and model training. The output of this project should be a tensorflow face recognization model, the model reaches ~ 95% accuracy in lfw (to be improved...)

So cut the crap, let's train some model!

# data preparing
We use CASIA-webface as our training dataset (contains 494,414 images,10575 individuals), other dataset( like microsoft's MS-Celeb-1M) can also be used as training dataset, althrough the result could be slightly different.

The download links of the CASIA-webface are listed as followed:
1. Through offical website: [http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html)
2. Through baidu cloud [https://pan.baidu.com/s/1nvtn6bf](https://pan.baidu.com/s/1nvtn6bf)

please be weare download from the offical website is always a more formal way.

after the data is downloaded ,it should be placed at data/CASIA-WebFace dir.

# use your own dataset
CASIA-Webface have a directory structure like this:
```
- CASIA-WebFace
|--- preson1
   |----pic1
   |----pic2
   |---- ...
|--- preson2
   |----pic1
   |----pic2
   |---- ...
```

Any dataset have the similar structure should be okay. Put them in data directory, and modify the configure INPUT_DIR in the  preprocess_data.py and you are good to go.

# data preprocessing
Data preprocessing is very important.
We use a cascade boxing method to find the faces in an image, and a cascade segmenter to align the faces found. 2d align has been implemented, however currently is not used becaused there is no evidence that 2d align have any advantage in our case. So what we do here is just to find the largest face in every picture, and save the face in a directory.

There is basically nothing you have to worry about, just run:

```
python3 preprocess_data.py 
```

And you shall get all the data we need

The data processed will be restored in data dir

# model training
After all the data is prepared, we now can train our face recognization model, however, we do not train a model directly to determine whether two face belongs to the same preson, instead, we train a model to determine exactly what preson is in the picture. In CASIA-WebFace's case, it's a multi classification problem of appromixately 10,000 classes, which is much difficult problem.

Our model is written on tflearn and tensorflow 1.2.0, we use a structure of 34 layer res-net, run
```
python3 train_classifier.py res-cp
```
Some other model are available in the script,too.
To train the model, it takes appromixately half a day and appromixately 20 epochs to train in a 1080ti gpu. After training, you should get a approximately 75 to 80 precent of accuracy

# validate on lfw
Face recognizetion algroithms are often validated on LFW to prove their power, you can download lfw dataset from the following links:
1. offical website: [http://vis-www.cs.umass.edu/lfw/](http://vis-www.cs.umass.edu/lfw/)
2. baidu cloud (already processed 6000 pairs) [https://pan.baidu.com/s/1bpjJy8n](https://pan.baidu.com/s/1bpjJy8n)

Download from the second link may save you a lot of trouble. After downloaded the 6000 pairs of the lfw, you should place them in lfw directory.
You may found a data_analysis.ipnb in the project dir, follow the codes there to find out how the remaining step of validate on lfw.

# Things that can be done but not yet try
1. Use center loss or triplet loss in the model to improve accuracy.
2. Use crops of face to train multi models and emsemble them.
3. Data argument(currently there are none).
4. Use bigger dataset.

All of these are very strong and promising techniques had been used and achieved great results. 

# requirements:
tensorflow 1.2.0
tflearn 
dlib for image preprocessing
cv2 for image preprocessing
numpy 
sklearn for data processing

# Contect
All questions and suggestions are welcome, please contect me at: icybee@yeah.net
