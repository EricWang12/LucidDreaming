# prompt="Two_different-sized_candles_arranged_in_a_line,_with_a_matchbox_to_the_left."
# prompt="Four_flower_pots_arranged_in_a_square,_with_a_garden_gnome_in_the_center."
# prompt="Three_cats_sitting_in_a_circle,_with_a_bowl_of_milk_in_the_center."
prompt="sitting_monkey"

method=DF
cls=chair

declare -A scene_dict
scene_dict[chair]="outputs/nerf-blender-old/chair@20230919-020850/ckpts/last.ckpt"
scene_dict[hotdog]="outputs/nerf-blender-old/hotdog@20230920-234745/ckpts/last.ckpt"
scene_dict[lego]="outputs/nerf-blender-old/lego@20230919-023118/ckpts/last.ckpt"
scene_dict[ship]="outputs/nerf-blender-old/ship@20230919-024827/ckpts/last.ckpt"
scene_dict[material]="outputs/nerf-blender-old/materials@20230919-023713/ckpts/last.ckpt"
scene_dict[mic]="outputs/nerf-blender-old/mic@20230919-024315/ckpts/last.ckpt"


echo  ${scene_dict[$cls]} 

directory=outputs/edit_blender/$cls/$method/$prompt/
latest_folder=$(ls -lt "${directory}" | grep '^d' | head -n 1 | rev | cut -d ' ' -f 1 | rev)

echo $directory$latest_folder
# echo outputs/multi_gen/magic3d/good_box/Four_flower_pots_arranged_in_a_square,_with_a_garden_gnome_in_the_center. 
# bash scripts/edit_blender/test.sh 1 $cls objects/edit/$cls/$prompt.txt $directory$latest_folder/ckpts/last.ckpt magic3d-blender
bash scripts/edit_blender/test.sh 1 $cls objects/edit/$cls/$prompt.txt  ${scene_dict[$cls]}  magic3d-blender
