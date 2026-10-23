# 高度源提取器：时间重复问题已修复，真实NetCDF合成回归通过

2026-10-23。本轮只做CPU输入工程准备；实际CPP和GHI未读，没有COT/GHI精度结果。上轮只有语法检查，本轮使用真正NetCDF4文件格式验证提取器的行为。

## 修正

旧extract_height_source_attempt1.py只检查8行且逐行时间不晚于init，没有强制唯一/连续；因此在某些重复源组合下可能把重复记录当作完整8帧历史。新版本在读取任何载荷前要求BJT下−105、−90、…、0分钟有序唯一的8时刻、源路径唯一、train作用域和UTC=BJT−8h，拒绝缺帧、重复、未来或错时区。原脚本保留，不修改此前几何数值或已完成GHI实验。

## 真实文件格式、合成内容的回归

在学院现有swc解释器下，用两只已校验SHA的CPU wheels建立独立.reader_deps_v1目录，只对本次子进程设置PYTHONPATH。netCDF4=1.7.2、cftime=1.6.4；复用原NumPy1.26.4，未改全局训练环境、未安装或更改CUDA/PyTorch，也未占GPU/PBS。

测试文件与临时grid全部自行生成，不复制或打开项目CPP/标签/test。八项检查如下：

- real_NetCDF4_packed_scale_units_descending_crop_fill_zero_QA_source_preservation：PASS
- duplicate_history_rejected：PASS
- nontrain_rejected_before_source_read：PASS
- wrong_UTC_rejected_before_source_read：PASS
- future_or_gap_rejected_before_source_read：PASS
- existing_output_not_overwritten：PASS
- swapped_dimension_names_rejected：PASS
- unmatched_geographic_grid_rejected：PASS

合成CTH包括packed int16的scale_factor、m/km两种单位、NaN缺测与有效零值，源LAT升序但目标grid降序；逐像元输出与预期最大差1.78e−15，缺测位置完全相同，源文件前后SHA不变。QA不存在时输出没有QA_raw且元数据明确False；不伪造全有效。维度名交换即拒绝，不根据正方形shape猜转置。

测试写入masked packed变量时netCDF4曾给出cast RuntimeWarning；已通过输出逐值、NaN和有效零值检查验证掩膜与数值保留。它来自合成夹具写入，不是提取真实源的警告或实际数据质量证据。

测试证明接口行为，不证明真实CPP高度单位、质量、历史填补、时间延迟或物理精度。SYNTHETIC_READER_TEST.json与READER_RUNTIME_RECEIPT.json保存代码/依赖哈希和运行状态，READER_RUNTIME_SPEC.json仍为事前声明，不能用声明代替完成收据。

## 实际访问与下一动作

达到上次退避时限后，重新只读连接FD107和FD201，两者仍Connection closed；ACCESS_RECHECK_20261023.json保留本轮证据。退避延长到30分钟，用户确认恢复可提前重试。不重建密钥、不把源访问问题称为数据缺失。

HEIGHT_SOURCE_REQUEST.json中的8份已绑定train历史源仍未提取。访问恢复后运行该提取器，先读取实际CTH属性/QA/坐标并检查时间，再决定是否能做物理投影；合成通过不等于真实源可用。未更新任何模型、原预测或论文，预报COT增益目标仍未达成。
