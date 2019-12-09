
source activate tensorflow_p36

pip install numpy scipy
pip install progressbar2
pip install scikit-learn

pip install gensim nltk
pip install chart_studio
python -m nltk.downloader punkt
python -m nltk.downloader stopwords

# Mount attached drive

suffix_path=$(lsblk | grep 50G)
str_arr=($suffix_path) # make into string array
full_path="/dev/${str_arr}"

# Format the volume to ext4 filesystem  using the following command
sudo mkfs -t ext4 $full_path

sudo mkdir /newvolume

sudo mount $full_path /newvolume/

cd /newvolume

# Need to grant permissions to ubuntu user to save and read files from mounted disk
sudo chown -R ubuntu /newvolume



#wget https://s3.amazonaws.com/mordecai-geo/GoogleNews-vectors-negative300.bin.gz
wget http://mattmahoney.net/dc/text8.zip

unzip text8.zip
rm text8.zip
cd ~/CCA-images-text
python ~/CCA-images-text/load_word2vec.py

pip install h5py

cd /newvolume

curl -O http://msvocds.blob.core.windows.net/coco2014/train2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip
unzip -q train2014.zip
unzip -q val2014.zip
unzip -q instances_train-val2014.zip
unzip -q captions_train-val2014.zip
rm train2014.zip
rm val2014.zip
rm instances_train-val2014.zip
rm captions_train-val2014.zip

pip install Cython
pip install pycocotools

cd $HOME
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make
python setup.py build_ext install

cd $HOME

git clone https://github.com/kevjp/CCA-images-text.git

# Copy across files that are too big to put on remote repo
scp -r -i /Users/kevinryan/Documents/DataScienceMSc/Rightmove/AWS/CompVisionLondon.pem /Users/kevinryan/Documents/DataScienceMSc/Rightmove/CCA_images_text/outputs ubuntu@ec2-3-8-174-95.eu-west-2.compute.amazonaws.com:/newvolume
scp -r -i /Users/kevinryan/Documents/DataScienceMSc/Rightmove/AWS/CompVisionLondon.pem /Users/kevinryan/Documents/DataScienceMSc/Rightmove/Results_google_images_resnet_classifiers/folder_2019-08-22/resnet_classifier ubuntu@ec2-3-8-174-95.eu-west-2.compute.amazonaws.com:/newvolume
scp -r -i /Users/kevinryan/Documents/DataScienceMSc/Rightmove/AWS/CompVisionLondon.pem /Users/kevinryan/Documents/DataScienceMSc/Rightmove/CCA_images_text/outputs/text.model.bin ubuntu@ec2-3-8-174-95.eu-west-2.compute.amazonaws.com:/newvolume

scp -r -i /Users/kevinryan/Documents/DataScienceMSc/Rightmove/AWS/CompVisionLondon.pem /Users/kevinryan/Documents/DataScienceMSc/Rightmove/Rightmove/GoogleImages/Fireplaces.zip ubuntu@ec2-3-8-174-95.eu-west-2.compute.amazonaws.com:/newvolume

scp -r -i /Users/kevinryan/Documents/DataScienceMSc/Rightmove/AWS/CompVisionLondon.pem /Users/kevinryan/Documents/DataScienceMSc/Rightmove/Rightmove/GoogleImages/Hardwood_Floors.zip ubuntu@ec2-3-8-174-95.eu-west-2.compute.amazonaws.com:/newvolume

scp -r -i /Users/kevinryan/Documents/DataScienceMSc/Rightmove/AWS/CompVisionLondon.pem /Users/kevinryan/Documents/DataScienceMSc/Rightmove/Rightmove/GoogleImages/Kitchen_Islands.zip ubuntu@ec2-3-8-174-95.eu-west-2.compute.amazonaws.com:/newvolume

scp -r -i /Users/kevinryan/Documents/DataScienceMSc/Rightmove/AWS/CompVisionLondon.pem /Users/kevinryan/Documents/DataScienceMSc/Rightmove/Rightmove/GoogleImages/Skylights.zip ubuntu@ec2-3-8-174-95.eu-west-2.compute.amazonaws.com:/newvolume

scp -r -i /Users/kevinryan/Documents/DataScienceMSc/Rightmove/AWS/CompVisionLondon.pem /Users/kevinryan/Documents/DataScienceMSc/Rightmove/Rightmove/ADE20K_tagged.zip ubuntu@ec2-3-8-174-95.eu-west-2.compute.amazonaws.com:/newvolume

unzip -q Fireplaces.zip
unzip -q Hardwood_Floors.zip
unzip -q Kitchen_Islands.zip
unzip -q Skylights.zip
unzip -q ADE20K_tagged.zip


