# Visual Token Matching — protótipo Taskonomy

Implementação didática e reduzida do artigo **Universal Few-shot Learning of
Dense Prediction Tasks with Visual Token Matching**. O experimento usa as
quatro partes centrais do VTM (encoder de imagem, encoder de rótulo, matching
e decoder) para aprender tarefas binárias definidas pelas classes semânticas
do Taskonomy.

O objetivo é produzir um experimento reproduzível em Colab/Kaggle, e não
reproduzir a tabela completa do artigo.

## Instalação

No Colab, use o PyTorch já instalado:

```bash
pip install -e ".[experiment]"
```

Para desenvolvimento local:

```bash
pip install -e ".[experiment,test]"
```

## Dados

### Execução local

O YAML padrão usa `data/taskonomy`. Faça primeiro um dry-run e depois baixe
apenas RGB e segmentação:

```bash
sudo apt-get install aria2

omnitools.download rgb segment_semantic \
  --components taskonomy --subset tiny \
  --dest ./data/taskonomy \
  --connections_total 16 --agree_all \
  --name "SEU NOME" --email "SEU_EMAIL" --dryrun

omnitools.download rgb segment_semantic \
  --components taskonomy --subset tiny \
  --dest ./data/taskonomy \
  --connections_total 16 --agree_all \
  --name "SEU NOME" --email "SEU_EMAIL"
```

> **Nota sobre `omnidata-tools==0.0.23`:** essa versão pode terminar com
> `KeyError: 'omnidata'` logo depois de exibir a licença do Taskonomy, pois falta
> uma entrada no mapa interno de licenças. O notebook Colab aplica
> automaticamente uma correção de compatibilidade no runtime antes do dry-run.
> Para executar pelo terminal, prefira a célula de download do notebook.

Se o dataset já estiver em outro local, altere apenas `data.root` em
`configs/taskonomy_vtm.yaml`. Confira os pré-requisitos:

```bash
vtm-taskonomy doctor --config configs/taskonomy_vtm.yaml
```

Depois execute cada etapa somente quando a anterior terminar com sucesso:

```bash
vtm-taskonomy prepare --config configs/taskonomy_vtm.yaml
vtm-taskonomy train --config configs/taskonomy_vtm.yaml
vtm-taskonomy evaluate --config configs/taskonomy_vtm.yaml
```

No Colab, o notebook troca `data.root` para
`/content/drive/MyDrive/taskonomy` automaticamente.

Um teste completo com dados sintéticos, sem download e sem pesos externos:

```bash
vtm-taskonomy smoke --output-dir outputs/smoke
```

O notebook [`notebooks/vtm_taskonomy_colab.ipynb`](notebooks/vtm_taskonomy_colab.ipynb)
contém o fluxo recomendado para Colab.

## Saídas

`evaluate` grava em `experiment.output_dir`:

- `results.csv` e `summary.csv`;
- `hypothesis.json`;
- painéis de predição em `panels/`;
- mapas de atenção em `attention/`.

Cada linha registra método, classe, shots, seed, IoU, Dice, precisão, recall e
taxa de falso positivo. A hipótese pré-registrada é avaliada comparando VTM e
baseline nos regimes 5-shot e 10-shot.

## Diferenças em relação ao artigo

- classes one-vs-rest são usadas como tarefas;
- o encoder visual ViT-Tiny fica congelado;
- adaptação usa quatro vetores de bias, um por nível;
- resolução de treino é 128×128;
- o label encoder e o decoder são menores que os originais.

As diferenças são intencionais para manter o experimento didático e viável em
uma única GPU de Colab.
