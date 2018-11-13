DATA_DIR=data
mkdir -p $DATA_DIR
wget http://ai.stanford.edu/~btaskar/ocr/letter.data.gz -O $DATA_DIR/letter.data.gz
gzip -d $DATA_DIR/letter.data_new.gz
