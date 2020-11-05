from core.parse_voc_test import ParsePascalVOC
from configuration import PASCAL_Test_DIR


if __name__ == '__main__':
    voc = ParsePascalVOC()
    voc.write_data_to_txt(txt_dir=PASCAL_Test_DIR)
