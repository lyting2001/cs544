# gear_rat_cfp


* Start by Setting Up the Environment
* * install pip requirements
"pip install -r ./utils/requirements.txt"

* Now see if everything runs on some movie review data
* * train a rationale model
python train_hugrat.py configs/example_rationale_configs.json

* * train counterfactual predictors
python train_GAN_CFPredictor.py configs/example_CFPredictor_configs.json
python train_GAN_CFPredictor.py configs/example_CFPredictor_configs2.json

* * dump the counterfactuals
python dump_CFs_dyn.py ../models/gancfp/ train.dyn.dump -td train -flipit 1 -iterdecode 1 

* * make a new training file
python make_CF_file.py ../models/gancfp/train.dyn.dump 0 -flipit 1 


* Now do everything with your own data
Hopefully everything ran through on the IMDb data. Now, you need to run those same scripts with your own datasets. The data should be formated so that the train, dev, and test splits are in separate files. Each line in the file has the class (0 or 1) a tab character (\t) and then the text. Look at the IMDb data files as examples.
After you've set up the data files, you need to modify the argument jsons. The rationale model config for IMDb was "example_rationale_configs.json". Copy this file so we can modify it for your data. The "log_path" field determins where the model is saved, so you should modify this so its different than the IMDb model. Now change the train_file, dev_file, and test_file fields to point to your new data files. You can also change source_train, source_file if you like. Now that you have a new config file, use that to train a new rationale model.
You've now trained the rationale model and it should be sitting on disc. We now need to train a to_positive and a to_negative counterfactual predictor model. You need to make a config file for each. You can start with "example_CFPredictor_configs(2).json" files. The "to_class" field control which class is generated. The "log_path" field again says where the model is saved, so you need to change this again. Change the data path fields as you did with the rationale config file. In the rationale subfiled of the CFP config, you need to change the "load_chkpt_dir" path to point to the log_path of the rationale model you just trained. This field is in {"rationale":{"load_chkpt_dir":"new_path"}. DO NOT change the "load_chkpt_dir" path in the head json ({"load_chkpt_dir":}). Now train the two seperate CFPredictors with the two config files.
You now have both sides of the CF predictor. You now need to dump the counterfactuals to file. Use the dump_CFs_dyn script as before but chang the first field to point to your new CF Predictor log_path. You can then make a new training file make_CF_file as before.