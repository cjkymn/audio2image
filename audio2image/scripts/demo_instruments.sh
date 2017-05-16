. CONFIG

checkpoint_dir=${CHECKPOINT_DIR} \
img_dir=${INST_IMG_DIR} \
net_gen=normalized_vec_700_net_G.t7 \
net_txt=${INST_NET_TXT} \
audiofeatures=scripts/audiofeatures.txt \
dataset=instruments \
th aud2img_demo_simp.lua
