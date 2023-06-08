# This code is used for stylizing images


# This code is used for multiple photo stylizing with one model
command="python stylize_arg.py"

images_dir="/home/xzzit/code/dataset/Nerf/real_world/mountain_1/baseline"
output_dir="./stylized_images"
imgs=($(find "$images_dir" -type f -name "*.jpg"))

for img in "${imgs[@]}"
do
    $command --c "$img" \
             --m "./4_cont1p0E05_sty1p0E10_cons1E00_tv1E00.pth" \
             --output-path "$output_dir" \
             --output-name "$(basename "$img")"
done