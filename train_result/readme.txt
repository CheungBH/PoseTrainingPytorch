1. Training
(1) Use the excel in [train_result/alphapose/training_demo.csv] to design s series of models.
(2) Copy the path of the last folder to the [config.py]
(3) Use "train_result/csv2cmd.py" to transfer the excel into commands.
(4) Use "train_result/customized_cmd.py" to assign cuda number (-1 refers to not assigned).
    The commands will be generated in [train_result/"your folder"/tmp.txt]
(5) Copy the commands to "auto_training.py". Run it.


2. Analysing
After training:
(1) The comprehensive training results will be written in [result/'device'_'your folder'.csv]. It can be opened by excel.
(2) The result of each model will be gathered in [exp/"your folder"/"each ID"/"each ID"].
(3) The models are stored in [exp/"your folder"/"each ID"/"each ID"].
(4) Use [train_result/collect_result.py] can gathered the results of each model. The results will be in [result/"your folder"]


3. Testing on video
(1) Copy the target model, and put it in Pose_with_CNN.
(2) Modify the class_num and model path in the file [config/config.py]


-------------------------------------------------------------------------------------------------------------------

4. What can be analysed of the results 
a) Which model achieves the highest [val_acc] or best performance considering speed and accuracy
b) Under which configurations will the models perform better or worse?
c) The models will perform bad in which kind of videos? And whether there are enough training data in this situation? 