# 异常画像（B 题宏基因组数据）

> 阶段 0.4 产出 | 方法：Isolation Forest + LOF（k=20）| 样本 484

## 1. 异常检测结果

| 方法 | 判据 | 标记异常数 |
|---|---|---|
| Isolation Forest | contamination='auto'（偏移 -0.5） | 0 |
| Isolation Forest（敏感性） | contamination=0.05 | 25 |
| LOF | k=20，LOF > 2 | 211 |

- **Isolation Forest 'auto' 标记 0 个**：IF 得分集中在窄带 [-0.447, -0.301]，'auto' 偏移 -0.5
  低于全部得分 → 无全局离群。说明数据在 IF 意义下**全局同质**，无强全局异常。
- **LOF（k=20，LOF>2）标记 211 个（43.6%）**：局部密度离群
  占比高，反映 92% 稀疏数据的固有特性（多数样本在局部邻域密度不均），非数据错误。
- **高置信异常（两法同时标记，严格口径）= 0 个**：因 IF 'auto' 为 0，严格交集为空。
- **敏感性高置信集（IF cont=0.05 ∩ LOF>2）= 25 个**：作为候选异常清单（见下）。

## 2. 候选异常清单（IF cont=0.05 ∩ LOF>2，偏离 >2σ 的 Top 特征）

| 样本索引 | dataset | disease | 偏离特征（z 值） |
|---|---|---|---|
| 4 | Zeller_fecal_colorectal_cancer | n | Enterococcus|Enterococcus_gallinarum (z=+22.0)、Carnobacterium|Carnobacterium_maltaromaticum (z=+22.0)、Enterococcus|Enterococcus_gilvus (z=+18.4) |
| 6 | Zeller_fecal_colorectal_cancer | n | Porphyromonas|Porphyromonas_gingivalis (z=+22.0)、Lactobacillus|Lactobacillus_vaginalis (z=+21.9)、Lactobacillus|Lactobacillus_salivarius (z=+21.1) |
| 17 | Zeller_fecal_colorectal_cancer | cancer | Treponema|Treponema_denticola (z=+22.0)、Alloprevotella|Alloprevotella_rava (z=+22.0)、Catonella|Catonella_morbi (z=+21.9) |
| 18 | Zeller_fecal_colorectal_cancer | cancer | Campylobacter|Campylobacter_curvus (z=+22.0)、Acidaminococcus|Acidaminococcus_sp_D21 (z=+21.9)、Clostridium|Clostridium_clostridioforme (z=+21.4) |
| 21 | Zeller_fecal_colorectal_cancer | n | Atopobium|Atopobium_minutum (z=+22.0)、Lactobacillus|Lactobacillus_antri (z=+22.0)、Streptococcus|Streptococcus_mutans (z=+18.7) |
| 23 | Zeller_fecal_colorectal_cancer | cancer | Actinomyces|Actinomyces_sp_HPA0247 (z=+22.0)、Parascardovia|Parascardovia_denticolens (z=+21.6)、Atopobium|Atopobium_rimae (z=+21.2) |
| 25 | Zeller_fecal_colorectal_cancer | cancer | Filifactor|Filifactor_alocis (z=+22.0)、Prevotella|Prevotella_nigrescens (z=+21.9)、Mogibacterium|Mogibacterium_sp_CM50 (z=+21.4) |
| 27 | Zeller_fecal_colorectal_cancer | cancer | Eremococcus|Eremococcus_coleocola (z=+22.0)、Corynebacterium|Corynebacterium_jeikeium (z=+22.0)、Mobiluncus|Mobiluncus_unclassified (z=+22.0) |
| 29 | Zeller_fecal_colorectal_cancer | cancer | Desulfotomaculum|Desulfotomaculum_ruminis (z=+22.0)、Clostridium|Clostridium_hylemonae (z=+21.9)、Eggerthella|Eggerthella_sp_1_3_56FAA (z=+21.6) |
| 30 | Zeller_fecal_colorectal_cancer | small_adenoma | Actinomyces|Actinomyces_graevenitzii (z=+21.9)、Bilophila|Bilophila_unclassified (z=+20.5)、Desulfovibrio|Desulfovibrio_piger (z=+20.4) |
| 34 | Zeller_fecal_colorectal_cancer | n | Corynebacterium|Corynebacterium_amycolatum (z=+22.0)、Brevibacterium|Brevibacterium_massiliense (z=+22.0)、Abiotrophia|Abiotrophia_defectiva (z=+21.8) |
| 38 | Zeller_fecal_colorectal_cancer | cancer | Aeromonas|Aeromonas_veronii (z=+22.0)、Leptotrichia|Leptotrichia_wadei (z=+22.0)、Actinomyces|Actinomyces_cardiffensis (z=+22.0) |
| 39 | Zeller_fecal_colorectal_cancer | cancer | Rothia|Rothia_unclassified (z=+20.6)、Klebsiella|Klebsiella_oxytoca (z=+20.5)、Stomatobaculum|Stomatobaculum_longum (z=+20.3) |
| 41 | Zeller_fecal_colorectal_cancer | cancer | Campylobacter|Campylobacter_rectus (z=+22.0)、Streptococcus|Streptococcus_oligofermentans (z=+21.5)、Porphyromonas|Porphyromonas_endodontalis (z=+19.4) |
| 47 | Zeller_fecal_colorectal_cancer | cancer | Bifidobacterium|Bifidobacterium_pseudolongum (z=+21.5)、Lactobacillus|Lactobacillus_acidophilus (z=+18.2)、Bifidobacterium|Bifidobacterium_bifidum (z=+16.7) |
| 51 | Zeller_fecal_colorectal_cancer | n | Anaerostipes|Anaerostipes_caccae (z=+21.8)、Erysipelotrichaceae_noname|Clostridium_ramosum (z=+19.8)、Coprobacillus|Coprobacillus_sp_29_1 (z=+19.2) |
| 67 | Zeller_fecal_colorectal_cancer | cancer | Clostridium|Clostridium_scindens (z=+12.7)、Clostridiaceae_noname|Clostridiaceae_bacterium_JC118 (z=+9.7)、Bacteroides|Bacteroides_nordii (z=+7.2) |
| 71 | Zeller_fecal_colorectal_cancer | small_adenoma | Alloscardovia|Alloscardovia_omnicolens (z=+21.9)、Streptococcus|Streptococcus_vestibularis (z=+17.9)、Streptococcus|Streptococcus_infantis (z=+15.9) |
| 74 | Zeller_fecal_colorectal_cancer | cancer | Peptostreptococcus|Peptostreptococcus_unclassified (z=+22.0)、Streptococcus|Streptococcus_intermedius (z=+21.9)、Peptoniphilus|Peptoniphilus_lacrimalis (z=+21.8) |
| 76 | Zeller_fecal_colorectal_cancer | small_adenoma | Streptococcus|Streptococcus_macedonicus (z=+21.9)、Streptococcus|Streptococcus_gallolyticus (z=+21.9)、Clostridium|Clostridium_perfringens (z=+19.8) |
| 88 | Zeller_fecal_colorectal_cancer | cancer | Candida|Candida_tropicalis (z=+22.0)、Nakaseomyces|Candida_glabrata (z=+22.0)、Anaerostipes|Anaerostipes_sp_3_2_56FAA (z=+22.0) |
| 110 | Zeller_fecal_colorectal_cancer | n | Pediococcus|Pediococcus_lolii (z=+22.0)、Prevotella|Prevotella_multisaccharivorax (z=+19.3)、Pediococcus|Pediococcus_unclassified (z=+19.0) |
| 204 | metahit | ibd_ulcerative_colitis | Streptococcus|Streptococcus_constellatus (z=+20.6)、Fusobacterium|Fusobacterium_nucleatum (z=+10.8)、Alistipes|Alistipes_sp_AP11 (z=+9.9) |
| 207 | metahit | n | Lactobacillus|Lactobacillus_animalis (z=+22.0)、Lactobacillus|Lactobacillus_helveticus (z=+22.0)、Kingella|Kingella_unclassified (z=+22.0) |
| 223 | metahit | ibd_ulcerative_colitis | Peptoniphilus|Peptoniphilus_sp_oral_taxon_375 (z=+22.0)、Actinomyces|Actinomyces_europaeus (z=+22.0)、Porphyromonas|Porphyromonas_somerae (z=+21.6) |

## 3. 分组关联

- 候选异常 dataset 分布：Chatelier_gut_obesity:0、Zeller_fecal_colorectal_cancer:22、metahit:3
- 候选异常 disease 分布：cancer:13、ibd_ulcerative_colitis:2、n:7、small_adenoma:3
- 候选异常所在簇：簇0:16、簇1:9

## 4. 数据质量关联

- 若候选异常在某一数据集/疾病上高度集中，提示该分组可能需独立建模或存在数据采集差异。
- 本数据异常多为「某物种异常高丰度」的稀疏离群，属宏基因组数据常见现象，非数据错误。
- 偏离特征 z 值普遍高达 +20 以上，是 StandardScaler 对 92% 稀疏数据的固有放大效应（稀有物种
  在少数样本中检出时标准化 z 值被拉高），非真实 20σ 偏离；1.4 阶段 CLR 变换可缓解。
- 结论：**无强全局异常，无需回退 0.3 清洗**；局部离群在 1.4 阶段可经 CLR 变换 + 正则化自然吸收。
