An nnU-Net network (https://github.com/MIC-DKFZ/nnUNet) is most optimally trained without dropout. 
However, for computing the uncertainty along with it's predictions with Monte-Carlo dropout (https://doi.org/10.48550/arXiv.1506.02142) it is necessary to perform dropout during inference. 

To be able to perform MC-dropout on a train nnU-Net network, it is therefore necessary to insert dropout layers (i.e. dropout surgery). 
When adapting two files (/nnunet/utilities/get_network_from_plans.py and /nnunet/inference/predict_from_raw_data.py) it becomes possible to perform dropout during inference. 

In the file /nnunetv2/utilities/get network from plans.py, the variable ’dropout op kwargs’ was altered from 0 to the desired dropout rate for the key ’PlainConvUNet’ in the dictionary ’kwargs’. 

As dropout was not implemented during training, a change in the network structure had to be made in order to insert the dropout layers. This was done in the file nnunetv2/inference/predict from raw data.py. 
In the function predict logits from preprocessed data, in the state dictionary (’self.network’) were modified by replacing all instances of ”all modules.1.” to ”all modules.2.” in all parameter keys (’params.keys()’). This adjustment was necessary as the dropout layer was inserted
before the renamed layers. After renaming the parameters, the model’s weights were updated.
