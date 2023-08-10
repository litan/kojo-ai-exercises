### Gen-art with Neural Style Trtansfer

This project involves using Neural Style Transfer (NST) to create generative-art.

Quick steps:
* Open any one of the supplied kojo scripts (within Kojo).
  * The script ending in `_gh` uses an NST model from Github.
  * The script ending in `_th` uses an NST model from Tensorflow Hub.
* Specify your style and content images in the script.
* Run the script to generate your art.
  * Note - the first run can take some time - anywhere from 5 to 30 seconds depending on the speed of your computer. Subsequent runs will be faster. Just wait for the output to appear after you run the program. The Kojo *Stop* button in the script editor toolbar will be active while your program is running (this can be used as an indicator that the computer is working).
  * If a run takes more than 15 seconds, Kojo will complain about a potential deadlock. You can safely ignore this *error*.


Models and images for this project are from:
https://github.com/emla2805/arbitrary-style-transfer
https://www.tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization

Some images are from:
https://github.com/magenta/magenta/tree/main/magenta/models/arbitrary_image_stylization
