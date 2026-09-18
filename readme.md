# plant_model —— 甘蔗株高识别（占位，未开工）

这个文件夹留给**识别甘蔗株高**的模型。目前是空的，只有这份说明。
旁边 [../root_model](../root_model/readme.md) 是已经做完的**根系分割 / 根长统计**项目，两者是平级的两个项目。

```
Deep_learning_model_for_sugarcane\
├── root_model\       根系模型（已完工，自带 git 仓库）
└── plant_model\      株高模型（本目录，待开工）
```

## 开工前要先定的三件事

### 1. 株高从哪种图算？

现有素材**全是俯拍**（5472×3648，相机固定、从上往下），来源两处：

| 位置 | 数量 | 说明 |
| --- | --- | --- |
| `E:\baiduwangpan\DownLoad\总图\plant\` | 1736 张 | 217 编号 × 4 重复 × 2 个日期（20251116 / 20251126），命名 `root_*-*_<日期>ST.jpg` |
| `root_model\datasets\root\` | 36 张 | 人工标注过的子集，命名 `plant_*-*_<日期>ST.jpg` |

两边**不是同一份文件**（同名图 md5 不同、字节数也不同，像是分别导出/压缩过），
所以第一件事是确认**哪一份算原始**，以及株高到底能不能从俯拍图里读出来
（俯拍看得到整株平铺的样子吗？还是必须另拍侧视图？）。

### 2. 标注口径

- 株高量的是**茎基部 → 最高叶尖**，还是**茎基部 → 生长点/茎顶**？（两者差很多）
- **定标**：怎么把像素换算成毫米？根系项目里 `config.MM_PER_PX` 至今是 0
  （= 只输出像素），株高如果要有物理单位，必须先量一个已知尺寸。
- 谁标、标多少张、要不要留一部分做测试集（按**植株**划分，别按图片划分 ——
  这是 root_model 踩过的坑，见 [../root_model/readme.md](../root_model/readme.md) 的「数据说明」）。

### 3. 任务形式

| 方案 | 做法 | 代价 |
| --- | --- | --- |
| **A. 端到端回归** | 输入图 → 直接输出一个高度数值 | 最省标注（一张图一个数），但可解释性差、标尺一变就得重训 |
| **B. 分割 + 量长度** | 先分割出植株（茎/叶），再按骨架量像素长度 | 标注贵（要画掩码），但**能直接复用 root_model 的全套基础设施**，且和根系统计同一套口径 |

**倾向 B**：root_model 已经把「三通道分割 → 骨架化 → 量长度 → 出 CSV」这条链路跑通了，
株高只是换一个量测对象，`common/` 里大部分东西能直接搬。

## 能从 root_model 直接搬的东西

| 模块 | 能复用吗 |
| --- | --- |
| [common/image_io.py](../root_model/common/image_io.py) | ✅ 图像读写、缩放、掩码上采样，与任务无关 |
| [common/naming.py](../root_model/common/naming.py) | ✅ 时间戳命名、并发安全的目录创建 |
| [common/ckpt.py](../root_model/common/ckpt.py) | ✅ 权重存取、按文件/目录解析模型 |
| [common/unet.py](../root_model/common/unet.py) | ✅ 网络结构（改 `out_ch` 即可） |
| [common/dataset.py](../root_model/common/dataset.py) | ⚠️ 骨架能复用，但真值掩码的画法要重写（根系折线 → 植株轮廓） |
| [train/train.py](../root_model/train/train.py) | ⚠️ 训练循环、早停、学习率调度、日志格式可整段搬；损失函数按任务换 |
| [inference.py](../root_model/inference.py) | ⚠️ 流程可搬，输出内容要换 |
| [tool/](../root_model/tool/) | ✅ 大多是通用工具（`separate_dataset` / `repair_*` / `add_suffix` 等） |

> 搬的时候**别直接复制整个文件夹**，那会把根系专用的东西（RSML 解析、根长统计、
> 检查范围那一套）也带过来变成包袱。按上表挑着抄。

## 约定（与 root_model 保持一致，避免两套习惯）

- 文件名格式 `plant_<编号>-<重复>_<日期><批次>.jpg`，解析时先剥前缀再去空格；
- 输出按**时间戳**建文件夹，重名追加 `-1`、`-2`（`naming.create_unique_dir`）；
- 训练/验证**按植株整组划分**，绝不让同一株的不同时点分处两侧；
- 数据不入库（`.gitignore` 里忽略 `datasets/`），服务器上单独同步。
