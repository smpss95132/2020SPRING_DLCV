# TODO: create shell script for running your improved UDA model

# Example
wget "https://www.dropbox.com/s/n43rwf7t3lcm5zj/%5B4%5DUSPS2MNISTM.bin?dl=1"
wget "https://www.dropbox.com/s/agzx65vw4ht9xw9/%5B4%5DSVHN2USPS.bin?dl=1"
wget "https://www.dropbox.com/s/rhpx21l093pytkm/%5B4%5DMNISTM2SVHN.bin?dl=1"
python3 p4.py $1 $2 $3
