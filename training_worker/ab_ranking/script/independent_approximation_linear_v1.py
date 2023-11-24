import os
import torch
import sys
from datetime import datetime
from pytz import timezone

base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from utility.regression_utils import torchinfo_summary
from training_worker.ab_ranking.model.ab_ranking_linear import ABRankingModel
from training_worker.ab_ranking.model.reports.ab_ranking_train_report import get_train_report
from training_worker.ab_ranking.model.reports.graph_report_ab_ranking import *
from data_loader.ab_ranking_dataset_loader import ABRankingDatasetLoader
from training_worker.ab_ranking.model.reports.get_model_card import get_model_card_buf
from utility.minio import cmd
from training_worker.ab_ranking.model import constants
from training_worker.ab_ranking.model.reports import score_residual, sigma_score
from data_loader.dataset_phrase_vector_loader import PhraseVectorLoader


def train_ranking(dataset_name: str,
                  minio_ip_addr=None,
                  minio_access_key=None,
                  minio_secret_key=None,
                  ):
    # load phrase
    phrase_loader = PhraseVectorLoader(dataset_name=dataset_name,
                                       minio_ip_addr=minio_ip_addr,
                                       minio_access_key=minio_access_key,
                                       minio_secret_key=minio_secret_key)
    phrase_loader.load_dataset_phrases()
    phrase_loader.upload_csv()


def test_run():
    train_ranking(dataset_name="propaganda-poster",
                  minio_ip_addr=None,  # will use defualt if none is given
                  minio_access_key="nkjYl5jO4QnpxQU0k0M1",
                  minio_secret_key="MYtmJ9jhdlyYx3T1McYy4Z0HB3FkxjmITXLEPKA1",
                  )


# if __name__ == '__main__':
#     test_run()
