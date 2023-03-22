#!/bin/bash

helpFunction()
{
	echo ""
	echo "Usage: $0 -p csvs_path"
	echo -e "\t-p path to unfiltered csv files"
	exit 1 # Exit script after printing help
}

while getopts "p:" opt
do
	case "$opt" in
	p ) p="$OPTARG" ;;
	esac
done

# print helpFunction in case parameters are empty
if [ -z "$p" ]
then
	echo "Did not provide a path with csv files";
	helpFunction
fi

echo "python 1EuroFilter.py --path <>"

mkdir $p/1euro

for filename in $p/*.csv
do
	echo ${filename}
	python 1EuroFilter.py --path $filename
done
