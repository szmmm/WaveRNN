### env
source activate p37_pt11_c9_tts

### gpu
# export CUDA_VISIBLE_DEVICES=0,1

### dir
EXP_DIR=/home/dawna/tts/qd212/models/WaveRNN
cd $EXP_DIR

### exp
## default: Taco + NV
# voc_weights=/home/dawna/tts/qd212/models/WaveRNN/quick_start/voc_weights/latest_weights.pyt
# voc_weights=${EXP_DIR}/checkpoints/lj_pretrain.wavernn/latest_weights.pyt
# python train_tacotron.py
# python train_wavernn.py --gta
# python gen_tacotron.py griffinlim
# python gen_tacotron.py wavernn --voc_weights $voc_weights
# python gen_wavernn.py -s 3 # --gta

## ASNV
# python train_wavernn.py --hp_file scripts/hparams_asnv.py
# python gen_wavernn.py --hp_file scripts/hparams_asnv.py -s 3 # --gta

## pretrain / init
# hp_file=scripts/hparams_init.py
# python train_wavernn.py --hp_file $hp_file --gta --init_weights_path ${EXP_DIR}/checkpoints/ljspeech_mol_asnv.wavernn/wave_step1000K_weights.pyt
# python gen_wavernn.py --hp_file $hp_file -s 3 --unbatched --gta

# gold init
hp_file=scripts/hparams_initGold.py
# python train_tacotron.py --hp_file $hp_file
# python train_wavernn.py --hp_file $hp_file --gta
# python gen_wavernn.py --hp_file $hp_file -s 3 --unbatched --gta

## debug
hp_file=scripts/hparams_debug.py
# python train_tacotron.py --hp_file $hp_file
# python train_wavernn.py --hp_file $hp_file --gta --init_weights_path ${EXP_DIR}/checkpoints/ljspeech_mol_asnv.wavernn/wave_step1000K_weights.pyt

tts_weights=/home/dawna/tts/qd212/models/WaveRNN/quick_start/tts_weights/latest_weights.pyt
# voc_weights=/home/dawna/tts/qd212/models/WaveRNN/quick_start/voc_weights/latest_weights.pyt
voc_weights=${EXP_DIR}/checkpoints/lj_pretrain.wavernn/latest_weights.pyt
# python gen_tacotron.py --hp_file $hp_file --tts_weights $tts_weights griffinlim
python gen_tacotron.py --hp_file $hp_file --tts_weights $tts_weights wavernn --voc_weights $voc_weights
# python gen_wavernn.py --hp_file $hp_file -s 3 --voc_weights $voc_weights # --gta
