import os
import gdown
import tarfile

# - Set project directory
topdir= os.getcwd()
print("topdir=%s" % (topdir))

############################
##   DATASET INFO
############################
dataset_name= "galaxy_mnist-dataset"
dataset_dir= os.path.join(topdir, dataset_name)
dataset_filename= 'galaxy_mnist-dataset.tar.gz'
dataset_url= 'https://drive.google.com/uc?export=download&id=1OprJ_NQIFyQSRWqjGLFQsAMumHvJ-tMB'
filename_train= os.path.join(dataset_dir, "train/1chan/datalist_train.json")
filename_test= os.path.join(dataset_dir, "test/1chan/datalist_test.json")
filename_train_3chan= os.path.join(dataset_dir, "train/3chan/datalist_train.json")
filename_test_3chan= os.path.join(dataset_dir, "test/3chan/datalist_test.json")

#################################
##      DOWNLOAD FILES
#################################
def download_files_from_gdrive(url, outfile, force=False):
  """ Download file from Google Drive """

  if force or not os.path.isfile(outfile):
    try:
      print(f"Starting download: {outfile}")
      gdown.download(url, outfile, quiet=False)
      print(f"Download completed: {outfile}")
    except Exception as e:
      print(f"Error downloading {outfile} from {url}: {e}")
  else:
    print(f"File {outfile} already exists. Skipping download.")
   
def untar_file(filename, extract_to='.'):
  """ Unzip tar.gz file to the specified directory """
  try:
    with tarfile.open(filename, 'r:gz') as tar:
      tar.extractall(path=extract_to)
      print(f"Extracted {filename} to {extract_to}")
  except (tarfile.TarError, FileNotFoundError) as e:
    print(f"Error extracting {filename}: {e}")


if __name__ == "__main__":
  # - Enter top directory
  os.chdir(topdir)

  # - Download dataset
  print("Downloading file from url %s ..." % (dataset_url))
  download_files_from_gdrive(url = dataset_url, outfile = dataset_filename, force=False)
  print("DONE!")

  # - Untar dataset
  print("Unzipping dataset file %s ..." % (dataset_filename))
  untar_file(dataset_filename)
  print("DONE")
