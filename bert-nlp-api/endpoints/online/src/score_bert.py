import os
import logging
import json
import numpy
import torch
from bert_model import BertModelForLivedoor
from preprocess import text_to_loader
from transformers import BertJapaneseTokenizer, BertModel


def init():
    """
    デプロイ時に実行される関数
    """

    # BERTトークナイザー、学習済みモデル、自作モデル
    global tokenizer
    global pretrained_bert 
    global model

    tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    pretrained_bert = BertModel.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", output_attentions=False, output_hidden_states=False)
    model = BertModelForLivedoor(pretrained_bert)

    # AZUREML_MODEL_DIRはデプロイ時に作成される環境変数（yamlで指定）
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model/single_bert_fine_tuning_livedoor.pth"
    )
    print("Loading model ...")
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print("... Complete.")
    logging.info("Init complete")


def run(raw_data):
    """
    エンドポイントが呼び出されると実行される関数
    """
    logging.info("model 1: request received")
    data = json.loads(raw_data)["data"]

    # データローダに変換
    dataloader = text_to_loader(data, tokenizer)

    # モデルの推論モードに切り替え
    model.eval()

    # GPUが使えるならGPUにデータを送る
    batch = next(iter(dataloader))
    print(f"Batch Size: {batch['ids'][0].size()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    inputs = batch["ids"][0].to(device)  # 単語ID列

    # 推論の実行
    print("Infer is processing ...")
    outputs = model(inputs)
    print("... Complete.")
    
    _, preds = torch.max(outputs, 1)  # ラベルを予測
    print(f"preds: {preds}")

    preds_num = preds.to('cpu').detach().numpy()
    print(f"preds_num: {preds_num}")

    # レスポンスデータ
    results = []
    for i, p in enumerate(preds_num):
        res = {}
        # res["text"] = batch["text"][i][0:20] + "..."
        # tmp["pred"] = current_app.config["ID2LABEL"][p]
        res["pred_label"] = str(p)
        results.append(res)

    # results = json.dumps(results, ensure_ascii=False)

    logging.info("Request processed")
    return {"results": results}
