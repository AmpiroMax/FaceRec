import logging
import os
import sys
import tarfile

import wget
from google_drive_downloader import GoogleDriveDownloader as gdd

RAW_DATA_PATH = "data/raw/"

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
    stream=sys.stdout
)

logger = logging.getLogger("Data downloader")


def download_LFW():
    logger.info("Creating folder for data")
    os.mkdir(RAW_DATA_PATH + "LFW/")

    logger.info("Downloading data from official site")
    url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    wget.download(url, "data/raw/LFW/")

    logger.info("Unpacking data")
    file = tarfile.open(RAW_DATA_PATH + "LFW/lfw.tgz")
    file.extractall(RAW_DATA_PATH + 'LFW/')
    file.close()
    logger.info("Successfully downloaded data!")


def download_CelebA():
    logger.info("Creating folder for data")
    os.mkdir(RAW_DATA_PATH + "CelebA/")

    logger.info("Downloading data from official site")
    # url = "https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=share_link&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ"
    # wget.download(url, "data/raw/LFW/")
    gdd.download_file_from_google_drive(
        file_id='0B7EVK8r0v71pZjFTYXZWM3FlRnM',
        dest_path=RAW_DATA_PATH + "CelebA/",
        unzip=True
    )


if __name__ == "__main__":
    download_CelebA()
