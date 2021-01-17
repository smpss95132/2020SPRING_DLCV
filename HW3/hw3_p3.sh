# TODO: create shell script for running your DANN model

# Example
wget "https://www.dropbox.com/s/hzu7pni1cp5esqp/USPS2MNISTM.bin?dl=1"
wget "https://www.dropbox.com/s/ifijqcjgsbh8bxx/SVHN2USPS.bin?dl=1"
wget "https://www.dropbox.com/s/dq7eftsxmgnzgnf/MNISTM2SVHN.bin?dl=1"
python3 p3.py $1 $2 $3
