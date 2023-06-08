# This code is used to hyperparameter tuning


# train with different weights
command="python train_arg.py"

contents=("1e0" "1.1e0" "1.2e0" "1.4e0")
styles=("2.12e4" "2.13e4" "2.14e4" "2.15e4")
cons_tv=("1e0" "3e0" "5e0"  "7e0")
batchsizes=("4" "8" "16" "24" "32")

# for style in "${styles[@]}"
# do
#     for content in "${contents[@]}"
#     do
#         $command --dataset /home/xzzit/Data/ImageData/CoCo/train2017/ \
#                 --style-image pretrained_models/Fauvism_André-Derain_Pier.jpg \
#                 --batch-size 32 \
#                 --style-weight "$style" \
#                 --content-weight "$content"
#     done
# done

for b in "${batchsizes[@]}"
do
    $command --dataset /home/xzzit/Data/ImageData/CoCo/train2017/ \
            --style-image pretrained_models/Fauvism_André-Derain_Pier.jpg \
            --batch-size "$b" \
            --model-name "$b"
done


# train with different style reference images
# command="python train_arg.py"
# params=($(find /home/xzzit/Data/ImageData/art/ -type f -name "*.jpg"))
# for param in "${params[@]}"
# do
#     $command --dataset /home/xzzit/Data/ImageData/CoCo/train2017/ \
#              --style-image "$param" \
#              --batch-size 32
# done


# train with different network architecture
# command="python train_arg.py"
# models = ('ae', 'bo', 'res', 'dense')
# for model in "${models[@]}"
# do
#     $command --dataset /home/xzzit/Data/ImageData/CoCo/train2017/ \
#              --model-name "$model" \
#              --batch-size 32
# done