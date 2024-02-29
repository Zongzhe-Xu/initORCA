CURPATH=`pwd`

# OTDD 
cd otdd
pip install -r requirements.txt
pip install .

# CONLL Features
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1P4xjZjyx2WKVOnZ7hocUq-ldA6vfKn2x' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1P4xjZjyx2WKVOnZ7hocUq-ldA6vfKn2x" -O text_xs_16.npy && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1A4YH1TdYt9xYtWBEpliBTo7ut-jwOAPl' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1A4YH1TdYt9xYtWBEpliBTo7ut-jwOAPl" -O text_ys_16.npy && rm -rf /tmp/cookies.txt

cd $CURPATH

