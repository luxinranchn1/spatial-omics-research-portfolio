# 数据、模型与方法来源

## 数据与基础模型

- [STP-Bench / STpredBench](https://github.com/NEXGEM/STpredBench)：公开数据入口与空间表达预测评测背景。本项目的少目标部署协议不是官方分数复现。
- [HEST公开数据入口](https://huggingface.co/datasets/MahmoodLab/hest)：病理与空间组学数据。遵守原始数据与模型访问条款。
- [TRIDENT](https://github.com/mahmoodlab/TRIDENT)：病理特征提取工具，包含原作者支持的模型接口。
- [scMMT](https://github.com/SongqiZhou/scMMT)：RNA–蛋白多模态模型，固定源码版本62ff3ab80eb8c8be59876d414038822c91063404；这里的归纳式适配与其原始转导流程不同。

## 采用的成熟方法

Ridge回归、k-center、TypiClust、知识蒸馏、部分标签监督、split conformal、CV+/jackknife+及协变量偏移加权校准均为已有方法，不作为本项目原创贡献。低秩求解是既定Ridge目标的计算实现，不改变统计目标。

第三方模型代码、原始图像/表达、权重未重新分发。引用和许可由各原作者仓库/论文管理，本项目代码许可不能替代其条款。
