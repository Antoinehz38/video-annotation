# Description

Pipeline to ease the labelling of a video using a tracker to automate annotation when the object doesn't change much between frames.

## Problematic

Labelling costs a lot in both time and money, and the data often has to remain within the company for confidentiality reasons.

When the goal is to label only a few videos, you might want a lightweight tool you can tune and adapt to your needs.

## My approach

### Idea

The goal is to semi-automate the data collection.  
Sometimes, the object keeps its orientation, size, and position in the video, which allows it to be tracked with a simple tracker.  
To ensure the bounding box fits the object accurately, it can be helpful to use a segmentation model.  
This way, you can click on the right bounding box before starting the tracking.  
For that reason, I chose to implement **FastSAM-s.pt**.

### Difficulty

Even though some trackers performed better than others (**MIL** and **KCF** worked best on my videos), the main issue is that they can't adapt the shape of their predictions — they always keep the same rectangle size.  
If the target changes size in the video, the label won't fit perfectly, leading to mislabelling or even re-labelling (worst case).

## Installation

I recommend creating a virtual environment and installing the dependencies using:

```bash
pip install .
```

You can download the SAM weights with:

```commandline
mkdir -p SAM_weight && wget https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-s.pt -O SAM_weight/FastSAM-s.pt
```

## Use

Use the [main.py](main.py) script by specifying your **video path**, the **type of tracker** you want, and the **path to the output dataset folder**.

You can expand the window to see better.

Instructions are displayed at the bottom of the window. Here is a short explanation:

1. There are three available modes:

   - **Navigation**: navigate in the video until you find what you need.  
     Press `n` for normal navigation or `t` for tracking.

   - **Tracking**: once ready, press `t` to activate tracking.  
     If SAM is activated, it will help you draw a bounding box.  
     If not, draw the bounding box manually and press **Enter**.  
     Press `b` to stop if the tracker starts making bad predictions (it will delete the last, wrong prediction).

   - **Normal**: label frames manually.  
     Draw a box on your target and press **Enter** to save and go to the next frame.  
     Press `b` to stop.

2. You can adjust several parameters using the trackbars:

   - **Frame trackbar**: navigate through the video.  
   - **FPS in tracking / 10**: sets the tracking speed.  
     The higher it is, the harder it becomes to stop when the tracker loses the target (you can change this during tracking).  
   - **Using SAM**: if set to 1, SAM will be called when tracking starts; otherwise, you must initialize the tracker manually.

## Results

I didn’t manage to complete a full pipeline — labelling a video with SAM + tracking and then training and testing a YOLO model — due to time constraints.


## Future work

It could be improved with additional features, such as:

- The ability to start tracking by simply clicking on the segmentation output instead of manually drawing a bounding box.
