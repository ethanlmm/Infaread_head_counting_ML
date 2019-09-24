# ir_ml
 infaread_head_counting_deep_learning


 archived:
    keras_train.py: loading process is nasty.
    model_bacup.py


data: IO/data reading function and path
model: some model
util: some preprocess function
main: train sequence




Comment:
    Feel free to change any thing. make it performs better!
    model:
     using maxpooling but didnt have upper sampling.
     So there need some shrink process in ground truth. I already done it and have saved the files in quater_xxx
     if you want to try some new model without using upper sampling, make a new data set first.

     lmm: I try to sampling each lairs from a sequential neural network and using dilation to process it.
     It may solve scale problem in video recording.
     ex: Person A stand from 100m away, person B stand from 70cm away. how to recognize both people properly.
     my solution is the filters in network may contain info like the whole face of A or the partial face of B.
     sampling every lairs will gave the ability to absorb info in different scale. But it just a prototype. performance is bad.
     it may because network using maxpooling.

    data:
     dont worry about the file names. as long as the name of csv file and picture match, it will work.
     You can download any dataset and try it. if the dataset contain different file types use regex to filter.
     for FOR function you can also try map instead.

    main: sgd would be a better choice since our data set is small.


using package:
Tensorflow-gpu2.0
onencv
numpy


