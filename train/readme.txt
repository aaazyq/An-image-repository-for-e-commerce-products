Simply running the run.ipynb can generate the submit result. It is on the google colab environment (but I suppose it can be run on other environments, too). 

I use two-stage training. The other notebooks generate intermediate csv results(they are all on the kaggle platform). I save those csv results as well as the model I created. 
second_stage_xgboost.ipynb uses those intermediate csv results to train a xgboost model for the final prediction. 

For other infomation, please see the pdf document.  