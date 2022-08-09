# AzureMchineLearningを使った機械学習(BERT)API

## 環境

## Docerのインストール
https://docs.docker.com/engine/install/

## ローカルにデプロイ
- 下記コマンドで既定の設定を確認できる
az configure -l -o table

- エンドポイント名の環境変数を設定
export ENDPOINT_NAME=bertendpoint

- ローカルにエンドポイントを作成
az ml online-endpoint create --local -n $ENDPOINT_NAME -f endpoints/online/managed/endpoint.yml

- ローカルのエンドポイントにデプロイ
az ml online-deployment create --local -n blue --endpoint $ENDPOINT_NAME -f endpoints/online/managed/nlp-blue-deployment.yml
※デプロイが完了するとinit()が自動で実行される