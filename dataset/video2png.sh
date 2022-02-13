folder="/data/dataset/video_decaptioning/train/X"
for filename in ${folder}/*
do  
    videoname=`basename ${filename%.*}`
    mkdir /data/dataset/video_decaptioning/train/imgs/X/"${videoname}"
    ffmpeg -i $filename /data/dataset/video_decaptioning/train/imgs/X/"${videoname}"/%03d.png
    fi
done