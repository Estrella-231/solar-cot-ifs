# 2026-11-01 性能诊断
用户要求提高生成吞吐。主PBS217013继续运行；独立PBS217014在空闲GPU4/5短测batch2/8/16。新文件build_bank_fast.py加入PHASE计时，移除已被dy_geo覆盖的未来legacy几何读取，其余输入/模型/25步/逐序列随机种子不变。新文件profile.pbs用于测吞吐与显存，生产代码build_bank.py保持原样。恢复模板resume_fast.template.pbs尚未执行，必须先通过实际吞吐、容量及对应输出比较。

恢复沿用相同分片索引和六个原始memmap；旧progress仅在数组flush后提交，恢复从最后已提交序列开始，保留已完成输出。新代码只接受由相同数据/权重/归一化、原生产者SHA导出的旧contract或本版本contract，保存原代码和恢复来源，不任意改写progress伪装原版本。
