from subprocess import call

child1 = "python prepare_data.py --split-file ../omg_TrainVideos.csv --target-dir ../Videos/Train"
call(child1.split(' '))

child2 = "python prepare_data.py --split-file ../omg_ValidationVideos.csv --target-dir ../Videos/Validation"
call(child2.split(' '))


child3 = "python prepare_data.py --split-file ../omg_TestVideos_WithoutLabels.csv --target-dir ../Videos/Test"
call(child3.split(' '))


