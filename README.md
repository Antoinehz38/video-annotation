# Description

Pipeline to ease the labellisation of a video using a tracker to automatize when the object doesn't change much between frames. 

## Problematic 

Labellisation cost a lot both in time and in money then the data has to remain in the company for some reason.

When the goal is just labelling few videos you might want a tool you can tune to adapt to your issues. 

## My approach

### Idea 
The goal is to semi automate the collect, sometimes the objet keep its orientation, size and region in the video thus allow 
the objet to be tracked with a simple tracker. Moreover, to be sure to fit the objet perfectly it could be nice to have a 
model that segment the image allowing you to just click on the right bounding box starting the tracking that is why i chose to implement SAM vitB. 

### Difficulty 

I don't know if that was my machine but the latency was very bad using the SAM model
(even the smaller one) and the prediction on my videos was not good (I think due to the quality mostly).
All combine made the use of SAM impossible. 

Also even if some tracker works better than others (MIL and KCF seemed to work on my video) the main issue is 
that they can't adapt their predictions shape always the same rectangle size so if the target change its size in the video
the label would not perfectly fit, which will result in mislabelling and maybe labelling a second time (worst case)

# Installation 

I advise to make a venv and install the dependency using

```commandline
pip install .
```

You can download the SAM weight using : 

```commandline
mkdir -p SAM_weight && wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O SAM_weight/sam_vit_b_01ec64.pth
```

# Use

Use [main.py](main.py) script by indicate your **video path**, the **type of tracker** you want and the **path of the saving dataset folder** 

Then you can expand the windows to see better

Indications are written at the bottom of the window to help you understand there some piece of explanation :

1. There is 3 modes available: 
   - **Navigation** : navigate in the video until you find what you are looking for. press 'n' for normal and 't' for tracking
   - **Tracking** : when find press 't' to activate tracking. It will call SAM if activated to help you draw a good bounding box. If not activated just draw a bounding box and press enter. press 'b' to break if the tracker make bad predictions (it will delete teh last prediction (which was wrong))
   - **Normal** : To just labellise the classical way. Draw a box on your target and press enter it will iterate the video. 'b' to break

2. There some parameters you can choose using the tracker bars:
   - **Frame track bar** to navigate in the video 
   - **Fps in tracking/10** to set the fps of the tracking the higher, the more difficult it will be to break when the tracker lose its target (you can change during tracking)
   - **Using SAM** if 1 SAM will be call when activate the tracking else you will initiate the tracker manually


# Results

I couldn't do a full pipeline: labelling a video only using SAM+Tracking and then training and testing a yolo, I didn't have enough time. 

I think that if one find a better Segment tools (smaller latency) it could be a useful tool

But I don't think that for now it's better than free open tools available.

# Futur work 

It could be improved easily with some features :

- Better segmentation tools (as we said) and like trackers the possibility to chose one (segmentor class from abstract class etc) 
- A clic just on the segmentation prediction instead of a drawing on 
