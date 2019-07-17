# AI_minor_OP4
Ai Minor period 4 - Deep Learning

To run realtime detection:
Set pythonpath:

  `export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim`

Run the python program from directory *'AI_minor_OP4/tensorflow_api'*:

  `python object_detection/detect_stream.py`


Run the python program over images *'AI_minor_OP4/tensorflow_api'*:

Put the images in 'tensorflow_api/test_images/' then run:

  `python object_detection/object_detection_runner.py`

The output of the images will be generated in 'tensorflow_api/output/'