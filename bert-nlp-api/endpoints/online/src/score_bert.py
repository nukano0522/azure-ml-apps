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
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """

    # BERTモデル
    global tokenizer
    global pretrained_bert 
    global model

    tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    pretrained_bert = BertModel.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", output_attentions=False, output_hidden_states=False)
    model = BertModelForLivedoor(pretrained_bert)

    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model/single_bert_fine_tuning_livedoor.pth"
    )
    print("Loading model ...")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print("... Complete.")
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
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
