# 已完成结果的阅读口径

| 文件 | 实验单位与范围 |
|---|---|
| model_annotation_2x2.csv | 乳腺癌6/肺癌14位目标患者；Ridge确定性，TRIPLEX3种子；200预算5位置；匹配源划分 |
| encoder_annotation.csv | 3癌种25位同一批患者；3编码器；50/100/200；各癌种基因端点不同 |
| selection_strategies.csv | 6策略与随机配对比较；含患者CI及合并/多重校正结果 |
| cancer_screening_flow.csv | 6癌种的公开数据可用性及患者筛查记录 |
| independent_roi_layout.csv | 额外14位胰腺癌患者，封存布局确认；不是新方法验证 |
| citeseq_classification.csv | 两迁移方向，5个监督/约束对照；3种子均值；固定9细类 |

所有文件从已有结果导出，不在导出过程中做新拟合或重新检验。Bootstrap等统计口径见各项目说明。不同源划分、癌种/基因端点和模型的绝对分数不混用。部分原始表有空值，按不可评价/不适用处理，不补零。
