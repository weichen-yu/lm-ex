#CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python hpo_hongyuan.py > output.log 2>&1 &
#
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1  simple_baseline.py --root-dir tmp/ --experiment-name baseline_3000/ --num-trials 3000  #tmux2
#CUDA_VISIBLE_DEVICES=6 python3 -m torch.distributed.launch --nproc_per_node=1  --master_port=1234 simple_baseline.py --root-dir tmp/ --experiment-name baseline/ --num-trials 1000  #tmux1
#CUDA_VISIBLE_DEVICES=7 python3 -m torch.distributed.launch --nproc_per_node=1  --master_port=1231 auto_parameters_baseline.py --root-dir tmp/ --experiment-name baseline/ --num-trials 100

#python3 simple_baseline.py --root-dir tmp/ --experiment-name test/ --num-trials 1000


#simple_baseline is weichen version
#baseline_orginigal is the google version
