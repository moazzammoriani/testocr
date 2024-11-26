# `testocr`

This is a tool I wrote to test the CER (Character Error Rate) for the following OCR libraries for handwriting detections. The testing is done on the [IAM](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) dataset.

- docTR
- PyTesseract
- EasyOCR
- keras-ocr
- PaddleOCR

## Usage
Here is an example demonstrate how to test the CER for `PyTesseract` on *line* images from the IAM dataset. `nforms` selects the number of *forms* from the IAM dataset. (In order to understand what *lines* and *forms* mean, it would be advisable to look up the structure of the IAM dataset.)


### Install dependencies
```
$ pip install -r requirements.txt

```

### Calling `Tester`
```python
from testocr import Tester
from helper.pytesseract import PytesseractOcr

if __name__ == "__main__":
    Tester.get_lines_cer(PytesseractOcr, nforms=10)
```

**Note: the function `Tester.get_lines_cer` assumes that your current directory contains the *lines* directory from the IAM dataset in your current directory (likewise, `Tester.get_words_cer` assumes you have the *words* directory in your current directory).**
