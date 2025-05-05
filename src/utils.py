import os
from pathlib import Path
import logging
import logging.config

# Create the Logger
logging.basicConfig(format="%(asctime)-15s %(levelname)s - %(message)s",datefmt='%Y-%m-%d %H:%M:%S')
logger= logging.getLogger("gmnist-classifier")
logger.setLevel(logging.INFO)

# - Set project directory
topdir= os.getcwd()
logger.info("topdir=%s" % (topdir))

############################
##   DATASET URL
############################
dataset_name= "galaxy_mnist-dataset"
dataset_dir= os.path.join(topdir, dataset_name)
dataset_filename= 'galaxy_mnist-dataset.tar.gz'
dataset_url= 'https://drive.google.com/uc?export=download&id=1OprJ_NQIFyQSRWqjGLFQsAMumHvJ-tMB'
filename_train= os.path.join(dataset_dir, "train/1chan/datalist_train.json")
filename_test= os.path.join(dataset_dir, "test/1chan/datalist_test.json")
filename_train_3chan= os.path.join(dataset_dir, "train/3chan/datalist_train.json")
filename_test_3chan= os.path.join(dataset_dir, "test/3chan/datalist_test.json")