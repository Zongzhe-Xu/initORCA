#!/bin/bash

# data download is automatic for CIFAR100 and MNIST, no need to run this script

DS="SPHERICAL DARCY-FLOW-5"

for i in $DS ; do
    if [ $i = 'SPHERICAL' ]
	then
		wget -c https://pde-xd.s3.amazonaws.com/spherical/s2_cifar100.gz
	
	elif [ $i = 'DARCY-FLOW-5' ]
	then
		wget -c https://pde-xd.s3.amazonaws.com/piececonst_r421_N1024_smooth1.mat
		wget -c https://pde-xd.s3.amazonaws.com/piececonst_r421_N1024_smooth2.mat
	
	elif [ $i = 'PSICOV' ]
	then
		mkdir protein
		cd protein
		wget -c https://pde-xd.s3.amazonaws.com/protein.zip
		unzip -a protein.zip
		rm protein.zip
		cd ..
	
	elif [ $i = 'COSMIC' ]
	then
		mkdir cosmic
		cd cosmic
		wget -c https://pde-xd.s3.amazonaws.com/cosmic/deepCR.ACS-WFC.train.tar
		mv deepCR.ACS-WFC.train.tar train.tar
		tar -xf train.tar
		wget -c https://pde-xd.s3.amazonaws.com/cosmic/deepCR.ACS-WFC.test.tar
		mv deepCR.ACS-WFC.test.tar test.tar
		tar -xf test.tar
		rm *.tar
		cd ..
		python3 preprocess_cosmic.py
	
	elif [ $i = 'NINAPRO' ]
	then
		mkdir ninaPro
		cd ninaPro
		wget -c https://pde-xd.s3.amazonaws.com/ninapro/ninapro_train.npy
		wget -c https://pde-xd.s3.amazonaws.com/ninapro/label_train.npy
		wget -c https://pde-xd.s3.amazonaws.com/ninapro/ninapro_val.npy
		wget -c https://pde-xd.s3.amazonaws.com/ninapro/label_val.npy
		wget -c https://pde-xd.s3.amazonaws.com/ninapro/ninapro_test.npy
		wget -c https://pde-xd.s3.amazonaws.com/ninapro/label_test.npy
		cd ..
	
	elif [ $i = 'FSD' ]
	then
		wget -c https://pde-xd.s3.amazonaws.com/audio/audio.zip
		unzip -a audio.zip
		rm audio.zip
	
	elif [ $i = 'ECG' ]
	then
		wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1j7Yk0-W9uonQiyyG2B9ZAw5jjZA-UeMS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1j7Yk0-W9uonQiyyG2B9ZAw5jjZA-UeMS" -O challenge2017.pkl && rm -rf /tmp/cookies.txt
	
	elif [ $i = 'SATELLITE' ]
	then
		wget -c https://pde-xd.s3.amazonaws.com/satellite/satellite_train.npy
		wget -c https://pde-xd.s3.amazonaws.com/satellite/satellite_test.npy
	
	elif [ $i = 'DEEPSEA' ]
	then
		wget -c https://pde-xd.s3.amazonaws.com/deepsea/deepsea_filtered.npz

	elif [ $i = 'MUSIC' ]
	then
		mkdir music
		cd music
		wget -c https://github.com/locuslab/TCN/blob/master/TCN/poly_music/mdata/JSB_Chorales.mat?raw=true -O JSB_Chorales.mat
		wget -c https://github.com/locuslab/TCN/blob/master/TCN/poly_music/mdata/Nottingham.mat?raw=true -O Nottingham.mat
		cd ..

	elif [ $i = 'FSD' ]
	then 
		wget -c https://pde-xd.s3.amazonaws.com/audio/audio.zip
		unzip -a audio.zip
		rm audio.zip
		pip install librosa
		apt-get install -y libsndfile1

	elif [ $i = 'Homology' ]
	then
		mkdir tape
		cd tape
		wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/remote_homology.tar.gz
		tar -xf remote_homology.tar.gz
		pip install tape_proteins
		cd ..

	elif [ $i = 'TinyImNet' ]
	then
		wget http://maxwell.cs.umass.edu/hsu/697l/tiny-imagenet-200.zip
		unzip tiny-imagenet-200.zip
		rm tiny-imagenet-200.zip

	elif [ $i = 'DomainNet' ]
	then
		mkdir domainnet
		cd domainnet

		wget http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip
		unzip -aq sketch.zip
		rm sketch.zip
		wget https://raw.githubusercontent.com/virajprabhu/SENTRY/main/data/DomainNet/txt/sketch_train_mini.txt
		wget https://raw.githubusercontent.com/virajprabhu/SENTRY/main/data/DomainNet/txt/sketch_test_mini.txt

		wget http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip
		unzip -aq real.zip
		rm real.zip
		wget https://raw.githubusercontent.com/virajprabhu/SENTRY/main/data/DomainNet/txt/real_train_mini.txt
		wget https://raw.githubusercontent.com/virajprabhu/SENTRY/main/data/DomainNet/txt/real_test_mini.txt

		wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip
		unzip -aq painting.zip
		rm painting.zip
		wget https://raw.githubusercontent.com/virajprabhu/SENTRY/main/data/DomainNet/txt/painting_train_mini.txt
		wget https://raw.githubusercontent.com/virajprabhu/SENTRY/main/data/DomainNet/txt/painting_test_mini.txt

		wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip
		unzip -aq clipart.zip
		rm clipart.zip
		wget https://raw.githubusercontent.com/virajprabhu/SENTRY/main/data/DomainNet/txt/clipart_train_mini.txt
		wget https://raw.githubusercontent.com/virajprabhu/SENTRY/main/data/DomainNet/txt/clipart_test_mini.txt


	elif [ $i = 'ListOps' ]
	then
		wget https://storage.googleapis.com/long-range-arena/lra_release.gz
		tar -xf lra_release.gz
		rm lra_release.gz 
		pip install datasets torchtext

	fi
done