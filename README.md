# DirectAfford-Experiment

公开的 DirectAfford 代码与实验整理仓库。当前提供统一的真实数据训练与评估入口，完整真实数据重训及官方基线对比核验尚未完成。

## 本次修复

- `src/train.py` 实际解析数据路径、epoch、随机种子、损失和输出目录。
- 优化器更新几何编码器及预测头，冻结文本编码器并保持其 eval 状态。
- 保留原始正样本对齐损失 `alignment`（默认）。新增 `bce` 和 `alignment_bce` 是背景监督实验变体，不能直接宣称为原论文方法。
- 每轮完整验证，默认以验证 MAE 保存 `best.pth`，可用 `--selection` 显式改变主选优指标；`last.pth` 包含优化器、配置、数据哈希、历史和随机状态，可恢复训练。
- 新评估入口强制严格加载权重、保留全部标注，报告 pooled 与 macro AUC/MAE。单类样本 AUC 为 null，仍计算 MAE。
- 旧 `eval*.py`、模拟数据基线训练脚本的命令行入口已停用。其定义保留用于审计，不是经过核验的实验 API。不要直接调用这些旧函数生成论文结果。

## 环境

```bash
conda env create -f environment.yml
conda activate directafford_env
```

Linux、Python 3.10、PyTorch 2.0.1/CUDA 11.8 是原环境配置。离线语义测试使用 CPU PyTorch；node107 的 PyTorch 2.0.1/CUDA 环境已完成真实数据单轮试跑，完整正式实验仍在进行。CLIP 通过 `--text-encoder` 指定本地目录或模型标识，默认 `openai/clip-vit-base-patch32`。模型使用冻结的 `CLIPTextModel.pooler_output`，并非官方 CLIP 投影后的跨模态特征；与其他模型对比时需要明确这一点。

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
- 训练、验证、测试 shape 必须互不重叠；独立测试入口检查训练和验证 shape ID。默认采样索引按 seed + 标注序号固定，点和 mask 同步；短点云保留原点并重复补足。可显式启用下述训练逐轮采样增强，验证和测试始终保持固定采样。LASO 坐标按官方流程中心化并缩放到单位球，退化点云报错。full-shape 坐标流程保持原样。不同预处理的基线不可直接混比。

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

### PyTorch 2.0.1 / NumPy compatibility

The original PyTorch 2.0.1 environment must use NumPy 1.x; `environment.yml` pins NumPy 1.26.4. If DataLoader reports `Could not infer dtype of numpy.float32`, first inspect versions and the NumPy bridge:

```bash
python -c "import torch, numpy as np; print(torch.__version__, np.__version__); print(torch.from_numpy(np.zeros(2, dtype=np.float32)))"
```

For an existing environment with NumPy 2.x, repair and rerun in a new Python process:

```bash
conda activate directafford_env
python -m pip install --force-reinstall --no-deps "numpy==1.26.4"
python -m pip check
python -m unittest discover -s tests -v
```

Passing the offline tests does not establish real-data accuracy. If the bridge still fails with NumPy 1.26.4, inspect the actual imported package paths and PyTorch installation before training.


## 双指标优化与配对筛选

目标是同一个 checkpoint 同时达到独立测试 AUC > 0.871、MAE < 0.089。阈值是优化目标，不是实测结果；旧表格的来源和评估协议仍待核验。开发只读取训练和验证数据，不在测试集上调参。

- 每轮分别保存 `best_auc.pth`、`best_mae.pth`、`best_joint.pth`。同一模型的指标成对记录，不拼接不同 epoch 的最佳数值。
- joint 分数为 `max(0,0.871-AUC)/0.871 + max(0,MAE-0.089)/0.089`。这是预先指定的折中选优规则，不能保证达标。达到目标时分数为0；严格是否超过目标由 `validation_targets_met` 单独报告。
- `best_feasible_auc.pth` 仅在验证 MAE < 0.089 时保存其中 AUC 最大的模型；文件存在不代表 AUC 已 > 0.871。
- `--alignment-weight` 和 `--bce-weight` 可控制组合损失；默认均为1，保持原配置。原始 alignment、纯 BCE、组合损失是不同实验设置。
- 每轮记录 alignment/BCE 分项、pooled 和 macro AUC、标签与预测均值、前景/背景平均预测值、学习率与耗时。每100 batch输出进度。
- `--scheduler plateau` 根据验证 AUC 停滞减半学习率；`--early-stop-patience` 默认为0（关闭），启用后依据 `--selection` 指标决定停止。
- 每个输出目录使用 Linux 文件锁，防止新版训练进程重复写入。旧版运行中的进程不持有这个锁，因此必须使用新目录。
- 旧 checkpoint 可在旧配置默认值下续跑，但无法恢复历史 AUC 最优模型。更换损失权重、scheduler 或选优配置需要另起独立训练，不改写正在运行的旧实验。

### Node107 四组50轮筛选

在GPU 1可用时执行；保留GPU 0上的旧任务。四组从同一种子42初始化、相同50轮预算、相同 plateau 调度，并按固定 joint 规则保存候选：
原始 alignment、纯 BCE、0.1×alignment+BCE、alignment+BCE。这一阶段是单种子验证筛选，胜出配置之后仍需要种子43、44复核。

```bash
cd /data/liwei/DirectAfford-Experiment
conda activate directafford_env
git pull --ff-only
export PYTHONNOUSERSITE=1
python -m unittest discover -s tests -v

DIRECTAFFORD_SCREEN="outputs/loss_screen_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DIRECTAFFORD_SCREEN"
nohup python -u scripts/run_loss_screen.py --gpu 1 --epochs 50 --seed 42 \
  --data-dir data/laso_dataset --output-root "$DIRECTAFFORD_SCREEN" \
  > "$DIRECTAFFORD_SCREEN/screen.log" 2>&1 &
echo "$DIRECTAFFORD_SCREEN"
```

脚本顺序执行四组，每组单独保存日志/权重，最后生成 `screen_summary.json`。重启时使用同一个 output-root 自动跳过已完成组并恢复未完成组；修改预算、种子、代码或数据路径必须用新根目录。第二次启动同一筛选根目录会被锁拒绝。测试集不参与筛选。

评估胜出配置之前冻结配置和选择规则，并补齐三种子重复；正式对比需统一测试样本、查询、归一化、点采样和 AUC 汇总方式。任何阈值是否达成最终以该冻结协议下的独立测试结果为准。


## 第二阶段：结构对照与验证内校准诊断

第一轮四组50轮筛选没有达到双指标目标。不要将局部结构修改直接当作已验证的性能提升；需要独立重训并在冻结协议下核验。

### 结构变体

`--architecture legacy`（默认）保持旧模型参数键、输出和 checkpoint 兼容。旧几何编码器是本仓库自定义模型，代码并未实际加载 Point-BERT 预训练权重。

`--architecture local` 在区域插值特征之外新增每点坐标和16邻域相对坐标的均值/最大值/最小值，将这12维描述编码到64维，再与384维区域特征融合。目的是检查区域插值是否限制了点级细节表达；不是既有方法已被证明的改进。

`--architecture query_local` 在 local 基础上，将冻结文本特征用于 FiLM 条件调制，再由小型预测头输出残差 logits，与原余弦 logits 相加。它改变了模型结构，是明确标注的实验变体。该分支必须使用含非零 BCE 权重的训练目标。

三组结构对照均使用纯 BCE、种子42、50轮、相同学习率调度和 joint 选优规则，分别为 legacy/local/query_local。不同时改变损失权重，避免把结构作用和损失作用混在一起。局部邻域增加计算和内存开销，真实运行耗时需在 node107 上测量。

### 校准诊断与查询敏感度

`src/evaluate.py --split val --audit-queries` 导出原始概率、标签、查询、shape ID和权重哈希，并报告替换查询后的平均预测变化。这个变化衡量敏感度，不是查询语义正确率。

`scripts/diagnose_calibration.py` 只接受未校准的 LASO 验证集导出，按 shape ID 分成拟合/检查两部分，拟合全局正斜率+偏置的 logit 映射，输出检查部分校准前后 AUC/MAE。默认拟合 MAE，可显式指定 BCE。不会使用测试标签、修改标签或删除困难样本。

**限制：checkpoint 已用完整验证集选优，随后划分的检查部分仍不是独立评估。** 校准诊断只能用于开发探索，不能当作无偏测试成绩。正斜率映射在数学上保持排序，不能提升排序能力；数值饱和可能产生额外平分，所以仍报告实际 AUC而不宣称严格数值不变。

校准 JSON绑定 checkpoint SHA256。正式使用前冻结权重、校准参数和选择协议，再在独立测试集一次性评估；用 `src/evaluate.py --calibration <calibration.json>` 会同时记录原始指标和校准指标，拒绝权重哈希不匹配的文件。

### Node107 一次启动第二阶段

```bash
cd /data/liwei/DirectAfford-Experiment
conda activate directafford_env
export PYTHONNOUSERSITE=1
git pull --ff-only
python -m unittest discover -s tests -v
nvidia-smi

DIRECTAFFORD_STAGE2="outputs/stage2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DIRECTAFFORD_STAGE2"
nohup python -u scripts/run_stage2.py \
  --source-screen outputs/loss_screen_20261014_041032 \
  --data-dir data/laso_dataset --gpu 1 --epochs 50 --seed 42 \
  --output-root "$DIRECTAFFORD_STAGE2" \
  > "$DIRECTAFFORD_STAGE2/stage2.log" 2>&1 &
echo "$DIRECTAFFORD_STAGE2"
```

确认GPU 1可用后启动。管线先对纯BCE/低权重组合的 best_joint 和原权重组合的 best_auc 做验证内校准诊断，再依次训练三种结构。数据、旧权重和代码均记录哈希；只读取 train/val，不评估 test。

输出包括 `stage2_plan.json`、`status.json`、`calibration_diagnostics/`、`structure/` 和最终 `stage2_summary.json`。日志每100 batch更新；保留第一轮全部结果。相同根目录重复启动会被锁拒绝，完整完成后重复调用不会重跑；恢复未完成任务需使用同一目录和冻结配置。

更新代码后不要恢复第一轮旧筛选管线，其源代码哈希已改变。新结构不能从旧几何模型自动续训，必须用新目录训练。现有旧 checkpoint 仍可由默认 legacy 模型严格加载和评估。

本阶段离线测试覆盖旧模型输出/权重兼容、局部和查询分支梯度、实际训练与评估调用、按对象校准划分、拒绝测试集拟合、校准权重绑定，以及第二阶段管线恢复。测试使用固定文本替代CLIP，仅验证语义，不证明真实精度提升。

## 第二阶段完成后的评估协议审计

官方来源固定为 [LASO c412ec3 的 eval_lyc.py](https://github.com/yl3800/LASO/blob/c412ec376310fd9d2de0d3ea06516dea50b9d4e8/eval_lyc.py) 和 [数据加载器](https://github.com/yl3800/LASO/blob/c412ec376310fd9d2de0d3ea06516dea50b9d4e8/data_utils/shapenetpart.py)。官方先按样本计算 AUC，再按样本汇总；标签用 `>=0.5`。当前训练选优使用 pooled AUC、标签 `>0.5`，不能把两种 AUC 混比。MAE始终用原始软标签。官方评估还进行3次模型投票，加载器直接使用对象的全部点；当前模型评估使用固定采样且一次预测。审计仅对同一份预测重新汇总，不声称完整复现官方协议。

`scripts/audit_protocol.py` 不训练、不拟合校准、不读取测试集。它读取已完成第二阶段的验证导出及三组完整历史，输出两种标签阈值下的 pooled/macro AUC、相同软标签 MAE、单类样本数、恰好0.5的标签点数和文件哈希。若样本没有正类或负类，AUC为未定义；审计显式计数并计算有定义样本的平均值。官方脚本在这类样本上可能产生NaN或报错，因此只有全部样本均有两类时，才能认为汇总规则对应。

历史汇总同时给出最高 pooled AUC、最高已有 strict macro AUC 和最低 MAE各自的完整轮次。最高 macro 对应轮次可能没有保存权重，不能把日志当作可独立评估的checkpoint。原训练及选优规则保持不变。

先运行无需GPU的历史/已有预测审计：

```bash
python scripts/audit_protocol.py \
  --stage2-root outputs/stage2_20261014_103937 \
  --output-dir "outputs/protocol_cpu_$(date +%Y%m%d_%H%M%S)"
```

确认GPU 1空闲后，可用一次命令顺序导出三组结构的 `best_joint.pth` 和 `best_auc.pth`，同时审计第一轮已有的三份验证预测。相同权重哈希只推理一次，最多6次结构验证，不重训：

```bash
DIRECTAFFORD_AUDIT="outputs/protocol_audit_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DIRECTAFFORD_AUDIT"
nohup python -u scripts/audit_protocol.py \
  --stage2-root outputs/stage2_20261014_103937 \
  --export-structures --data-dir data/laso_dataset --gpu 1 \
  --output-dir "$DIRECTAFFORD_AUDIT/results" \
  > "$DIRECTAFFORD_AUDIT/audit.log" 2>&1 &
echo "PID=$! 输出目录=$DIRECTAFFORD_AUDIT"
```

输出在 `results/protocol_audit.json`，逐个验证的日志在 `results/*.log`。使用新目录，禁止写入已完成的第二阶段目录；失败保留日志，再运行需新目录。审计不修改已有权重、历史、标签或阈值。校准导出和test导出会被拒绝。该工具对应的定向离线检查为 `python -m unittest discover -s tests -p test_protocol_audit.py -v`。

用户提供的截图包含 Point-BERT(0.871/0.136)、OpenAD(0.870/0.089)、LASO(0.858/0.098)、DirectAfford(0.860/0.092)。这些只是待核验的参考数值，截图未证明数据划分、模型权重、预处理和评估脚本一致。目标仍是冻结协议下同一模型同时 AUC>0.871、MAE<0.089；不得通过更换指标汇总方式来宣称模型精度提高。几何预训练仍未接入，当前自定义编码器与官方 Point-BERT权重不兼容，不能用 `strict=False` 把部分键加载成功视为完成预训练复现。

## 第三阶段：词级查询解码与同对象诊断

第二阶段已运行的三组联合选优结果（用户提供，种子42、验证集）为：legacy 0.8264/0.1040、local 0.8236/0.1018、query_local 0.8300/0.0994（pooled AUC/MAE），均未达标。协议审计中 query_local 的逐样本平均 AUC 为0.8363，MAE仍为0.0994。单一查询向量可能限制功能词与部位词的使用，这是待验证假设，不是已证明的瓶颈。

新增 `--architecture query_tokens`：保留 query_local 的局部特征、池化查询调制和余弦分支，在预测头前加入点特征到冻结 CLIP 最后一层文本 token 的6头交叉注意力、残差归一化和前馈层。padding token 被屏蔽，文本仍冻结。仅支持含非零 BCE 的目标；它是实验变体，不是官方 LASO 或 Point-BERT 复现。旧模型参数键与评分保持兼容，默认仍为 legacy。新变体需要从头训练，不将旧 checkpoint 部分加载冒充新结构训练。

`scripts/diagnose_queries.py` 提供两个用途：

- `--mode queries --checkpoint ...`：只读验证集。按 shape 找到不同 affordance、不同 mask 的查询对，每个对象选全点 mask 差异最大的一个对；最多128个对象。两条查询使用完全相同的坐标、点索引，并比较正确查询与互换查询对应预测的 MAE。报告正确查询误差更低的对象比例、互换前后 MAE 差及标签差异加权的预测方向。相比替换成任意别的查询，这个诊断能检验同一对象内的功能区分。它依然只描述这组特选样本，不能代表全验证集精度。
- `--mode overfit --architecture query_local|query_tokens`：只读训练集，固定最多8个对象的双功能查询及对应问题（Question1），分别从头训练300步。报告训练子集初始/最终 MAE、AUC与功能区分指标，作为容量和梯度诊断；不读取验证或测试，不是泛化成绩。输出的 `overfit_probe.pth` 明确标为诊断权重，不能当作正式训练 checkpoint。没有可用同对象双功能标注会报错，不捏造替代对。

配对采样按 shape ID哈希和种子确定，双查询共用点索引；它与正式验证逐标注采样不同。因此诊断分数不能与完整验证指标直接比较。单个对象的完整 mask 平均差必须>0.01才纳入，记录候选对象数量、选中标注索引、问题及采样后差异，不用预测成绩选择对象。

`scripts/run_query_stage.py` 一次顺序执行：已有 query_local 最优联合模型的同对象验证诊断、query_local 和 query_tokens 两个训练子集容量探针、两种结构各50轮的纯BCE配对训练。最后一步保持已有 pooled AUC/MAE联合选优及 plateau 调度，不因审计结果切换选优口径；同时保留每轮 macro AUC。容量诊断不设未经验证的自动过关阈值，正式两组都执行。目标是否达到仍须冻结配置后三种子重复与独立测试，不保证本轮胜出。

```bash
cd /data/liwei/DirectAfford-Experiment
conda activate directafford_env
export PYTHONNOUSERSITE=1
git pull --ff-only
python -m unittest discover -s tests -v
nvidia-smi

# 确认GPU 1空闲后，只启动一次。
DIRECTAFFORD_QUERY="outputs/query_stage_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DIRECTAFFORD_QUERY"
nohup python -u scripts/run_query_stage.py \
  --source-stage2 outputs/stage2_20261014_103937 \
  --data-dir data/laso_dataset --gpu 1 --epochs 50 --probe-steps 300 --seed 42 \
  --output-root "$DIRECTAFFORD_QUERY" \
  > "$DIRECTAFFORD_QUERY/stage.log" 2>&1 &
echo "PID=$! 输出目录=$DIRECTAFFORD_QUERY"
```

进度看 `status.json`、`stage.log`；同对象结果在 `same_object/diagnostic.json`；探针结果在 `overfit_pooled/diagnostic.json`、`overfit_tokens/diagnostic.json`；正式逐轮日志在 `screen/query_local_bce/train.log` 和 `screen/query_tokens_bce/train.log`。最终 `summary.json` 汇总诊断及选中的完整成对验证指标。所有新输出单独保存，不写入前两阶段目录。根目录加锁，配置/代码/数据/来源权重哈希绑定；同配置重跑跳过已完成诊断，正式训练由筛选脚本续跑，未完成的容量探针需检查后用新目录重跑。

离线检查增加了token屏蔽、有效token影响及梯度、文本冻结、同对象点标签配对、正确/互换查询对照、token模型真实训练与诊断调用，以及管线恢复/配置变更拒绝。使用固定文本替代CLIP；这些检查不证明真实词语语义或GPU精度。

## 第四阶段：训练采样增强与初始学习率的2×2对照

第三阶段用户报告的 query_tokens 最优联合验证结果为第16轮 pooled AUC=0.862926、macro AUC=0.852242、MAE=0.0908485；最低MAE轮次为第25轮，pooled AUC=0.851125、MAE=0.0897259。50轮内未达双指标目标。两种解码器在8个训练对象上的容量探针均可达到约0.989 AUC、0.023 MAE，这只证明能拟合该训练子集，不是泛化成绩。

新增 LASO 训练选项 `--train-sampling fixed|epoch`（默认fixed）和 `--train-keep-ratio`（默认1）。epoch模式的采样随机数由 `[seed, annotation_index, epoch]` 的 SeedSequence生成，与读取顺序和全局随机数无关。每轮开始设置epoch，续跑根据检查点轮次恢复相同采样；采样策略和保留比例加入严格恢复配置检查。旧检查点缺少这两项时按fixed/1兼容。

原始点数若正好等于2048，单纯重新采样/排列不会改变可见点集。因此本轮增强设置为epoch/0.85：先从完整归一化点云中无放回随机保留 `max(1,floor(0.85*N))` 个点，再按原配对采样规则采样或重复补齐到2048。坐标和soft mask始终使用同一索引，不引入噪声标签，不按前景位置选择点，不对稀疏子集重新归一化。增强同时改变了输入覆盖和重复点分布，可能改善也可能损害效果，必须对照验证。epoch/1仍可单独使用，但N等于目标点数时不会改变点集。

验证和测试强制使用fixed/1；训练点增强不会影响评估对象、点数、查询、标签阈值、聚合方式或选优规则。full_shape暂不支持该增强，显式请求会报错。`run.json` 记录原始训练点数范围、恰好等于目标点数的标注数量、增强策略和比例。

四组都从头训练query_tokens，纯BCE、seed42、batch16、2048点、50轮，使用相同plateau调度及原有pooled AUC/MAE联合选优，保留macro AUC。只改变如下两项：

| 目录名 | 训练采样 | 初始学习率 |
|---|---|---:|
| fixed_lr1e-4 | 原固定采样 | 0.0001 |
| epoch85_lr1e-4 | 逐轮85%点保留及配对补足 | 0.0001 |
| fixed_lr5e-5 | 原固定采样 | 0.00005 |
| epoch85_lr5e-5 | 逐轮85%点保留及配对补足 | 0.00005 |

固定采样1e-4组用于核验原结果在当前兼容代码下可复现。不要把不同轮次的最佳指标拼接；汇总中的macro AUC也是该选中轮次的指标。目标表的正式协议仍需统一，本轮不通过改变汇总口径来宣称精度提高。

在node107激活环境并通过测试后，确认GPU 1空闲，使用单条启动命令（无需手动设置输出变量）：

```bash
cd /data/liwei/DirectAfford-Experiment
conda activate directafford_env
export PYTHONNOUSERSITE=1
git pull --ff-only
python -m unittest discover -s tests -v
nvidia-smi
```

```bash
bash scripts/start_sampling_screen.sh 1 50
```

启动脚本使用当前Python环境，先检查指定GPU是否可用，再建立唯一的新目录；这不是GPU占用检测，请先查看nvidia-smi。启动后打印PID、绝对输出目录及可以直接复制的日志查看命令。四组顺序占用同一张GPU；不读取test文件，不写入前几轮目录。最终输出 `screen_summary.json`，每组的 `train.log` 每100批次和每轮更新，`history.json` 保留全部轮次；`screen_plan.json` 绑定配置、数据及Python源码哈希。

恢复未完成筛选时使用同一配置、同一根目录直接运行Python入口（不要再次运行启动脚本创建另一个任务）：

```bash
python -u scripts/run_sampling_screen.py --gpu 1 --epochs 50 --seed 42 \
  --data-dir data/laso_dataset --output-root <原输出目录>
```

仅在确认原任务已经退出后恢复。已完成组跳过，未完成组用last.pth恢复；更换配置、数据或代码必须另起目录。更新代码后不要恢复前几阶段冻结过源码哈希的旧管线；原权重和结果继续保留且可评估。

新增检查覆盖逐轮采样确定性、稀疏点与标签配对、验证输入保持固定、增强训练恢复与连续训练的参数完全一致、恢复配置变更拒绝，以及四组命令/已完成组跳过。离线测试通过不代表本轮真实精度已经提升。
