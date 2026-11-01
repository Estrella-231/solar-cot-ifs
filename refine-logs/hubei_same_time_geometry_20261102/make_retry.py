from pathlib import Path
p=Path(__file__).parent
s=(p/'run.pbs').read_text().replace('/run.log','/run_attempt2.log')
libs=['libcudnn_ops_infer.so.8','libcudnn_ops_train.so.8','libcudnn_cnn_infer.so.8','libcudnn_cnn_train.so.8','libcudnn_adv_infer.so.8','libcudnn_adv_train.so.8','libcudnn.so.8']
preload=':'.join('/public/home/slfu/miniconda3/envs/swc/lib/python3.10/site-packages/nvidia/cudnn/lib/'+n for n in libs)
s=s.replace('conda activate swc','conda activate swc\nexport LD_PRELOAD='+preload)
(p/'run_attempt2.pbs').write_text(s,newline='\n')
