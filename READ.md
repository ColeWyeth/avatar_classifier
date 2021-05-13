This project classifies characters from Avatar the Last Airbender using a CNN in Pytorch.
To train it you will need a training directory with subdirectories
named after the characters, e.g. Zuko, containing images with names of the form Zuko13.jpg etc. I originally had 16 training images and 4 test images per character (proof of concept) and if you have a different number you will need to change the dataset. Otherwise you can just load the .pth file and to use the pretrained model.  
 