# Pneumothorax Kaggle Competition

My project for detecting Pneumothorax on kaggle

https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview

Putting it here so I don't lose it.

Shoutout to my dad, Glenn Mackenzie, for helping me with understand the various chest X-Rays.

## What should I expect the data format to be?
The data is comprised of images in DICOM format and annotations in the form of image IDs and run-length-encoded (RLE) masks. Some of the images contain instances of pneumothorax (collapsed lung), which are indicated by encoded binary masks in the annotations. Some images have multiple annotations.

    1.2.276.0.7230010.3.1.4.8323329.14508.1517875252.443873,387620 23 996 33 986 43 977 51 968 58 962 65 956 70 952 74 949 76 946 79
    
Images without pneumothorax have a mask value of -1.

    1.2.276.0.7230010.3.1.4.8323329.1034.1517875166.8504,-1

##What am I predicting?

We are attempting to a) predict the existence of pneumothorax in our test images and b) indicate the location and extent of the condition using masks. Your model should create binary masks and encode them using RLE. Note that we are using a relative form of RLE (meaning that pixel locations are measured from the end of the previous run) as indicated below:

    1.2.276.0.7230010.3.1.4.8323329.14508.1517875252.443873,387620 23 996 33 986 43 977 51 968 58 962 65 956 70 952 74 949 76 946 79

Sample code is available for download that may help with encoding and decoding this form of RLE.

Some images may require multiple predictions - simply make each prediction its own separate line with the same ImageId.
