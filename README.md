# ニューラルネットワークの重み可視化

## 環境構築

```
conda create -n vismodel python=3.10
conda activate vismodel
pip install -e .[dev]
```

### Dockerでの環境構築

```
cd environment && docker compose up -d
```

起動後、コンテナ内で以下のコマンドを入力する

```
cd vismodel && conda activate vismodel && pip install -e .[dev]
```

### devcontainerでの環境構築
VSCodeのコマンドパレットを開く(Ctrl+Shift+P)

コマンドパレットからDev Containers: Rebuild and Reopen in Containerを指定