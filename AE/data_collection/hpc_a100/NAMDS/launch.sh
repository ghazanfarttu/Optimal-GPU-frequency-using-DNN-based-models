#!/bin/bash

numRuns=3
mode="dvfs"
#arch="GV100"
arch="GA100" 
oldfile="changeme"
#m_freq=877 #V100
#m_freq=1215 #A100
m_freq=1593 #A100 SXM
sleep_interval=1
t=0

declare -a progs=("namd") #executable name
declare -a apps=("NAMD") #any name
declare -a appParams=(" > results/NAMD_RUN_OUT") 

#declare -a freqs=(1380 1372 1365 1357 1350 1342 1335 1327 1320 1312 1305 1297 1290 1282 1275 1267 1260 1252 1245 1237 1230 1222 1215 1207 1200 1192 1185 1177 1170 1162 1155 1147 1140 1132 1125 1117 1110 1102 1095 1087 1080 1072 1065 1057 1050 1042 1035 1027 1020 1012 1005 997 990 982 975 967 960 952 945 937 930 922 915 907 900 892 885 877 870 862 855 847 840 832 825 817 810 802 795 787 780 772 765 757 750 742 735 727 720 712 705 697 690 682 675 667 660 652 645 637 630 622 615 607 600 592 585 577 570 562 555 547 540 532 525 517 510)

#declare -a freqs=(1380 510)

declare -a freqs=(1410 1395 1380 1365 1350 1335 1320 1305 1290 1275 1260 1245 1230 1215 1200 1185 1170 1155 1140 1125 1110 1095 1080 1065 1050 1035 1020 1005 990 975 960 945 930 915 900 885 870 855 840 825 810 795 780 765 750 735 720 705 690 675 660 645 630 615 600 585 570 555 540 525 510)

#declare -a freqs=(1410)

for c_freq in "${freqs[@]}"
do
	./control $m_freq $c_freq
	nRuns=$((numRuns-1))
	for i in $(seq 0 $nRuns)
	do
		appID=0
		for app in "${apps[@]}"
    	do	
			t=$((t+1))
			# GET RESULTS WITH NEW FREQUENCY
			echo "### $app - $c_freq (MHz) - Iteration: $i ###"
			file="results/$arch-$mode-$app-$c_freq-$i"
			
			#pushd apps
			#appPath=$(pwd)/
			#popd
					
			#appCmd="$appPath${progs[$appID]}${appParams[$appID]}"
			appCmd="./run_NAMD.sh${appParams[$appID]}"

      echo $appCmd
			./profile $appCmd

			cp $oldfile $file
			rm -f $oldfile
			sleep $sleep_interval

			# SAVE time for NAMD
			if [[ $app == "NAMD" ]]
			then
				#v=$(sed -n '258p' results/NAMD_RUN_OUT)
				v=$(grep "WallClock:" results/NAMD_RUN_OUT)
        tokens=( $v )
				wall_time=${tokens[1]}
				echo $wall_time
				wall_time="${wall_time//[$'\t\r\n ']}"
				printf '%s\n' $c_freq $wall_time | paste -sd ',' >> results/$arch-dvfs-namd-perf.csv
			fi
			appID=$((appID+1))
		done #apps
   	done #runs
done #freqs

echo "DONE!!!"

# revert the core frequency
#c_freq=1380 #V100
c_freq=1410 #A100
./control $m_freq $c_freq
echo $t
