from testocr import Tester
from helper.pytesseract import PytesseractOcr

if __name__ == "__main__":
    Tester.get_lines_cer(PytesseractOcr, nforms=10)

