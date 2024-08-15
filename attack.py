# Attack
import os
import json
import argparse
import openbackdoor as ob
from openbackdoor.victims.plms import PLMVictim
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results
import nltk
import OpenHowNet
from transformers import BertTokenizer, BertConfig, AutoModelForSequenceClassification
from openbackdoor.victims.custom_bert import CustomBertModel  # 确保这个导入路径正确
import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

OpenHowNet.download()

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/sos_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    print("Main function started")  # 添加调试信息
    # set_seed(config['seed'])

    # choose a victim classification model
    attacker = load_attacker(config["attacker"])
    victim = load_victim(config["victim"])

    # choose SST-2 as the evaluation data
    target_dataset = load_dataset(**config["target_dataset"])
    poison_dataset = load_dataset(**config["poison_dataset"])

    logger.info("Train backdoored model on {}".format(config["poison_dataset"]["name"]))
    backdoored_model = attacker.attack(victim, poison_dataset, config)

    if config["clean-tune"]:
        logger.info("Fine-tune model on {}".format(config["target_dataset"]["name"]))
        CleanTrainer = load_trainer(config["train"])
        backdoored_model = CleanTrainer.train(backdoored_model, target_dataset)

    logger.info("Evaluate backdoored model on {}".format(config["target_dataset"]["name"]))
    results = attacker.eval(backdoored_model, target_dataset)

    # 计算攻击成功率

    # asr = victim.calculate_asr()
    # sample_counts = victim.get_sample_counts()
    # logger.info(f"Sample counts: {sample_counts}")
    # logger.info(f"Attack Success Rate (ASR) on triggered samples: {asr * 100:.2f}%")

    display_results(config, results)



if __name__ == '__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)

    main(config)

