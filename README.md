# DirectAfford-Experiment

公开的 DirectAfford 代码与实验整理仓库。当前提供统一的真实数据训练与评估入口，尚未完成真实数据重训，也没有提供可核验的官方基线对比结果。

## 本次修复

- `src/train.py` 实际解析数据路径、epoch、随机种子、损失和输出目录。
- 优化器更新几何编码器及预测头，冻结文本编码器并保持其 eval 状态。
- 保留原始正样本对齐损失 `alignment`（默认）。新增 `bce` 和 `alignment_bce` 是背景监督实验变体，不能直接宣称为原论文方法。
- 每轮完整验证，以验证 MAE 保存 `best.pth`；`last.pth` 包含优化器、配置、数据哈希、历史和随机状态，可恢复训练。
- 新评估入口强制严格加载权重、保留全部标注，报告 pooled 与 macro AUC/MAE。单类样本 AUC 为 null，仍计算 MAE。
- 旧 `eval*.py`、模拟数据基线训练脚本的命令行入口已停用。其定义保留用于审计，不是经过核验的实验 API。不要直接调用这些旧函数生成论文结果。

## 环境

```bash
conda env create -f environment.yml
conda activate directafford_env
```

Linux、Python 3.10、PyTorch 2.0.1/CUDA 11.8 是原环境配置。本次离线语义测试使用 CPU PyTorch；原 CUDA 环境尚未实测。CLIP 通过 `--text-encoder` 指定本地目录或模型标识，默认 `openai/clip-vit-base-patch32`。模型使用冻结的 `CLIPTextModel.pooler_output`，并非官方 CLIP 投影后的跨模态特征；与其他模型对比时需要明确这一点。

## LASO 输入

LASO 官方来源：[yl3800/LASO](https://github.com/yl3800/LASO)。遵守发布方许可，pickle 和模型权重只从可信来源加载。

本仓库已对照 [LASO 官方数据加载器](https://github.com/yl3800/LASO/blob/main/data_utils/shapenetpart.py) 实现 query 和坐标处理，支持以下**明确的数据合同**；尚未用官方数据包验证，不保证原始 pickle 可直接读取。若格式不同，请先按官方字段转换，禁止用随机数据、全零点云或任意 mask 补缺。

```text
data/laso_dataset/LASO_dataset/
  anno_train.pkl       objects_train.pkl
  anno_val.pkl         objects_val.pkl
  anno_test.pkl        objects_test.pkl
  Affordance-Question.csv
```

- `anno_<split>.pkl`：list 或 dict 的 value 列表。每条含 `shape_id`、`affordance`（非空字符串或单元素字符串列表）、`mask`（长度 N 的 [0,1] 数组，或以完整 query 为键的字典）。多 query 标注须明确展开。
- `objects_<split>.pkl`：以 shape ID 为键的 dict，或每条带 `shape_id` 的 list；每个对象是坐标数组或只含一个坐标字段 `point_cloud` / `points` / `coordinate` 的字典。
- 官方 query 模式（默认）要求标注含 `class`，并按 CSV 的 `Object`/`Affordance` 精确匹配。训练使用 Question1–14 的随机改写，验证/测试使用 Question0；不再把 affordance 类名当成情境问题。`--laso-query-mode affordance` 仅供类名消融，不能称为官方 LASO 查询协议。
- 坐标为 `(N, >=3)`，取前三列；点与 mask 必须等长且有限。缺失 shape、歧义 query/mask、空点云立即报错。
- 训练、验证、测试 shape 必须互不重叠；独立测试入口检查训练和验证 shape ID。采样索引按 seed + 标注序号固定，点和 mask 同步；短点云保留原点并重复补足。LASO 坐标按官方流程中心化并缩放到单位球，退化点云报错。full-shape 坐标流程保持原样。不同预处理的基线不可直接混比。

## 真实数据训练与独立评估

先检查数据合同，再分别运行原始损失和背景监督变体。输出目录应独立，正式建议种子 42、43、44 分别重训。

```bash
python src/train.py --dataset laso --data-dir data/laso_dataset \
  --epochs 300 --seed 42 --loss alignment --output-dir outputs/alignment_seed42

python src/train.py --dataset laso --data-dir data/laso_dataset \
  --epochs 300 --seed 42 --loss alignment_bce --output-dir outputs/alignment_bce_seed42

python src/train.py --dataset laso --data-dir data/laso_dataset \
  --epochs 300 --seed 42 --loss alignment_bce --output-dir outputs/alignment_bce_seed42 \
  --resume outputs/alignment_bce_seed42/last.pth

python src/evaluate.py --checkpoint outputs/alignment_bce_seed42/best.pth \
  --data-dir data/laso_dataset --split test --output-dir outputs/eval_seed42
```

300 epoch 是启动默认值，不是经过验证的最优配置。脚本前台执行并输出每轮指标，后台运行请显式使用 `nohup`。`bash start_exp2_training.sh --loss alignment_bce` 仅训练 DirectAfford，不启动未经核验的基线。

新入口要求带配置和优化器状态的新格式 checkpoint；旧的纯 state_dict 不能自动续跑或直接进入新评估，应重新训练。

输出：`run.json`（配置及数据哈希）、`history.json`、`best.pth`、`last.pth`；评估输出 `metrics.json`、`per_sample.json`、`predictions.npz`。测试集不能用于选择损失、epoch 或温度；这些选择只在验证集完成。

## full-shape / 3D AffordanceNet

兼容已有 `full_shape_train_data.pkl` / `full_shape_val_data.pkl` 与三折类别协议：

```bash
python src/train.py --dataset full_shape --data-dir data --fold 0 \
  --loss alignment --epochs 300 --output-dir outputs/fullshape_fold0
python src/evaluate.py --checkpoint outputs/fullshape_fold0/best.pth \
  --data-dir data --split val --classes novel --output-dir outputs/fullshape_novel_fold0
```

训练 checkpoint 的选优只使用 val 文件中的 seen 类；novel 类评估单独执行。此仓库没有提供 full-shape 文件的下载或生成流程。需要额外核对训练/验证对象来源与独立性，不能把此协议当作 LASO 原始情境查询协议。

## 基线与旧表格的限制

现存 OpenAD、Aff3DFunc 训练函数使用随机演示数据，若旧权重由这些函数生成，不能称为真实数据训练权重。部分旧评估使用代理结构、随机输入、缺失权重继续运行或 `strict=False`。已有截图数值无法仅凭仓库定位其生成脚本，不能据此认定为真实官方对比。

`src/baselines/` 中三个 gitlink 缺少 `.gitmodules`，普通 clone 不会取得官方源码。当前没有擅自补猜测 URL；需核验原始 URL 与固定 commit，再恢复子模块和官方模型适配器。Point-BERT、OpenAD、LASO、Aff3DFunc 的正式对比仍需使用各自已核验的官方实现、兼容权重和同一测试协议。新入口目前只评估 DirectAfford，不生成这些方法的占位分数。

## 本地验证

```bash
python -m unittest discover -s tests -v
```

测试用固定的文本编码器替代 CLIP 下载，使用小型合成数据核验点/mask 配对、缺失数据报错、背景梯度、预测头更新、冻结文本、官方问题选择与坐标归一化、评估确定性、训练续跑及输出完整性。它们不验证真实 CLIP 语义、真实数据效果或 GPU 性能。
