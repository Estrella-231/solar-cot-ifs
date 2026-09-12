# CPP生产脚本扩大检索

用户指出应主动查找。本轮继续以当前账户只读搜索有权限的相关路径，未将查找工作直接退回用户；不读凭据、个人通信或绕过权限。未提交训练或更改任何源产品。

## 新增实际检查

- 公开Global_CPP/FY/FY4B/Model/COT/train-16y-batchsize128-100nan列有main.py、cot_best.pth、cot_end.pth、nohup.out、预测/真值npy与网络文件。已有main.py审计证明是训练/评价脚本，不能据目录邻接绑定Result/CPP生产。
- 公开SmaAT-Unet/Code四个AGRI属性notebook，Model/SmaATUnet为网络模块；FY4A根只有Model/Sample/Result。Global_CPP/Realtime本轮目录枚举超时，不能称为空或无writer。
- /home/Data_Pool_3/yangzx根可读，因此继续查找相关Read_Sat、SWC与三个notebook；SWC仅见SLWP_3d_slwp_100m_v2_20260825，未递归无关工程。
- /home/Data_Pool_3/yangzx/Code、Project以及/home/nvme/yangzx均实测PermissionError(13)，不是只根据mode700推断。没有权限变更或提权。
- Read.ipynb读取Global_CPP/Results/2020产品并绘图；MOD_CWP.ipynb读取MODIS云属性并计算CWP；均未找到目标Result/CPP writer。
- FY_Opt.ipynb大小2633442字节，首次2MB读取上限后提高到64MB有界读取；提取仅2500字符代码，没有CPP/COT/FY4/模型加载/NC写出命中。没有因notebook包含大量图像输出而把大小门槛误当权限限制。
- Read_Sat初查超时，后续成功读取FY4B/Read_FY4B.ipynb（167911字节，SHA aad238794a1530e6fda37a7485941322e5f4df5a9c9cc82371b66bd6d5c036a7）。它读取和比较CTH，未载入模型、未写NC。其新线索为/home/Data_Pool_3/yangzx/Data/FY/FY_CN-Data/CPP与FY_lihr/CPP，文件名均FY4B_AGRI_20250101003000.nc。路径名字不证明具体产品来源、算法或官方身份。
- 上述Data/FY及两个产品目录可枚举，两个产品目录只有CPP子目录；同级FY_Sys含FY4B_CPP_FD、Cbar，FY4B含AGRI/GIIRS/CBH。新路径尚不能绑定当前Global_CPP/FY/FY4B/Result/CPP的实际生成批次。

## 结论

可以且已自行扩大查找，新增了可读的产品比较代码和相关CPP来源路径；实际writer、checkpoint与归一化/批次绑定仍未确认。明确区分：私有目录是权限拒绝，Realtime是读取超时，已读notebook是消费者/计算分析代码。不能把“未找到”写成“服务器上不存在”。

后续若获得生成包或可读代码新路径，沿具体引用继续；不需要用户重新传整个数据集。空间表示训练仍受源绑定门禁，既有实验结论不改。

## 可复查产物

audits/cpp_expanded_search_20260910.jsonl、cpp_expanded_search_20260910_followup.jsonl、cpp_read_fy4b_notebook_20260910.json、cpp_related_products_20260910.jsonl。含代码SHA、相关代码片段、目录列表及原样异常，未把内容片段hash当整批NC provenance。

FY_Sys/FY4B_CPP_FD进一步浅层枚举15秒超时，未取得其子目录列表；不将目录名称当官方来源证明。
