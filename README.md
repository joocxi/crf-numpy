# Conditional Random Field
This is an implementation of simple CRF using NumPY. The model was trained and tested on OCR, a dataset for handwritten words.

## Dataset
http://ai.stanford.edu/~btaskar/ocr/letter.data.gz
## Usage
./download.sh <br />
python3 main.py

## Configurations
Train/Val/Test split: 5502/688/687 <br />
Learning rate: 0.001 <br />
Num epochs: 200

## Result
Validation accuracy: 83.81% <br />
Test accuracy: 83.29%


