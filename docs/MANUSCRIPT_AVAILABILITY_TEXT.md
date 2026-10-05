# Manuscript-ready availability statements

## English — anonymized initial submission

### Data Sharing Statement (full title page)

Data generated or analyzed during the study are available from the corresponding author by request. Access is limited to deidentified patient-level derived features and model risk scores and is subject to approval by the participating institutions, an appropriate data-use agreement, and a prohibition on re-identification or onward redistribution. Raw clinical records, CT images, segmentation masks, direct identifiers, and linkage keys will not be shared.

### Data Availability (main manuscript)

Individual-level clinical data, CT images, and segmentation masks cannot be made publicly available because the ethics approval and institutional data-governance conditions do not authorize unrestricted public disclosure, and complete de-identification of longitudinal imaging-linked records cannot be guaranteed. Deidentified patient-level derived features and model risk scores may be considered through controlled access after review of a research proposal, confirmation of ethics approval or exemption, approval by the participating institutions, and execution of a data-use agreement. Approved access will be limited to the minimum data necessary for the stated purpose; re-identification and onward redistribution will be prohibited. Aggregate results, variable definitions, and model specifications are included in the article and supplemental material.

### Code Availability (main manuscript)

An anonymized, patient-data-free review package containing the analysis code, frozen preprocessing parameters, trained model parameters, the self-supervised encoder weights, input schemas, and a reproducible inference example is provided as Supplemental Software for Review. The package excludes patient identifiers, individual follow-up times, outcomes, images, segmentation masks, and patient-level predictions. Upon publication, the version used for this study will be deposited in a publicly accessible Git repository and archived as an immutable release with a persistent DOI; the repository URL and unique revision identifier will be added to the article.

## English — replace after public release

### Code Availability

The analysis code, frozen preprocessing parameters, trained model parameters, self-supervised encoder weights, input schemas, and reproducible inference example are publicly available at `https://github.com/Xyh378652081/rectal-ct-rfs` (release `v1.1.0`). The public artifacts contain no patient identifiers, individual follow-up times, outcomes, CT images, segmentation masks, or patient-level predictions. The software release and repository link will be maintained for at least 5 years after publication.

## 中文核对稿

### 数据共享声明（完整标题页）

本研究生成或分析的数据可向通信作者提出申请。可申请的内容仅限去标识化的患者级衍生特征和模型风险评分；数据访问须经参与机构审批、签署适当的数据使用协议，并禁止重新识别或二次分发。原始临床记录、CT图像、分割标签、直接身份标识符及数据链接密钥不予共享。

### 数据可用性（主文）

由于伦理批准及各参与机构的数据治理要求不允许不受限制的公开披露，且与纵向随访信息关联的影像资料无法保证完全消除重新识别风险，因此患者级临床数据、CT图像和分割标签不能公开。去标识化的患者级衍生特征和模型风险评分可在受控访问框架下申请；申请者需提交研究方案，提供伦理批准或豁免证明，经参与机构审核并签署数据使用协议。获批数据仅限于实现所述研究目的所必需的最小范围，并禁止重新识别和再次分发。汇总结果、变量定义和模型说明已包含于正文及补充材料中。

### 代码可用性（匿名初投稿版本）

本研究随稿提供一份匿名且不含患者数据的审稿用软件包，其中包括分析代码、冻结的预处理参数、训练后的模型参数、自监督编码器权重、输入格式说明和可复现的推理示例。该软件包不包含患者身份信息、个体随访时间、结局、CT图像、分割标签或患者级预测结果。论文发表时，本研究所用版本将上传至公开Git代码仓库，并以具有永久DOI的不可变版本归档；届时将在论文中补充仓库地址和唯一代码版本标识。

### 代码可用性（公开仓库后替换）

本研究的分析代码、冻结的预处理参数、训练后的模型参数、自监督编码器权重、输入格式说明及可复现的推理示例已公开于 `[GITHUB_URL]`（版本 `v1.1.0`；提交号 `[COMMIT_SHA]`），并永久归档于Zenodo（`[ZENODO_DOI]`）。公开材料不包含患者身份信息、个体随访时间、结局、CT图像、分割标签或患者级预测结果。该归档软件及访问链接将在论文发表后至少维护5年。
