# prompt="Two_different-sized_candles_arranged_in_a_line,_with_a_matchbox_to_the_left."
# prompt="Four_flower_pots_arranged_in_a_square,_with_a_garden_gnome_in_the_center."
# prompt="Three_cats_sitting_in_a_circle,_with_a_bowl_of_milk_in_the_center."
prompt="Three_cats_playing_with_a_yarn_ball_in_a_circle."
# outputs/multi_gen/magic3d/extra/Three_different_sized_teddy_bears_arranged_in_a_line,_with_a_pink_ribbon_tied_to_the_middle_one./run@20231125-140217/ckpts/last.ckpt
method=magic3d
cls=complex

directory=outputs/multi_gen/$method/$cls/$prompt/
latest_folder=$(ls -lt "${directory}" | grep '^d' | head -n 1 | rev | cut -d ' ' -f 1 | rev)

echo $directory$latest_folder
# echo outputs/multi_gen/magic3d/good_box/Four_flower_pots_arranged_in_a_square,_with_a_garden_gnome_in_the_center.
bash scripts/multi_gen/test.sh 1 objects/multi_gen/$cls/$prompt.txt $method $directory$latest_folder/ckpts/last.ckpt