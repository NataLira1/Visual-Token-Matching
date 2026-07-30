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

> **Nota sobre `omnidata-tools==0.0.23`:** o pacote contém um módulo legado
> `omnidata_tools.starter_dataset` com imports inválidos. O notebook valida o
> módulo correto, `omnidata_tools.dataset.starter_dataset`, usado pelo comando de
> download. Ele também substitui a integração obsoleta com Google Forms por um
> aceite explícito dos links das licenças, sem transmitir nome ou e-mail.

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

### Colab com pouco espaço

Use [a configuração compacta](configs/taskonomy_vtm_compact.yaml) e o notebook
Colab. Esse modo baixa apenas o subconjunto Taskonomy `debug` (um prédio) para
`/content`, que é temporário, e persiste no Drive somente manifest, checkpoint e
resultados. Como há apenas um prédio, as imagens são divididas
deterministicamente por hash em 70% treino, 15% validação e 15% teste. Portanto,
o experimento compacto avalia generalização entre tarefas/classes, não entre
prédios. Como a cobertura varia por prédio, classes que não tenham imagens
suficientes para formar um episódio são registradas e ignoradas automaticamente;
o meta-treino exige pelo menos duas classes válidas. Pares RGB/máscara
corrompidos ou incompletos também são ignorados, com detalhes salvos em
`corrupt_records.json` ao lado do manifest.

No modo compacto do Colab, o dataset permanece em `/content/taskonomy`; não
desconecte ou reinicie o runtime entre download, preparação, treino e avaliação.

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
