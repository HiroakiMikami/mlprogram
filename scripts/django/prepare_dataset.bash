#! /bin/bash
set -u

PROJECT_ROOT=$(dirname $0)/../../

# Download raw dataset
echo "--- Downloading ---"
mkdir -p ${PROJECT_ROOT}/dataset/raw/django
mkdir -p ${PROJECT_ROOT}/dataset/tmp
git clone --depth 1 https://github.com/odashi/ase15-django-dataset/ ${PROJECT_ROOT}/dataset/tmp/django
cp ${PROJECT_ROOT}/dataset/tmp/django/django/all.anno dataset/tmp/django/django/all.code ${PROJECT_ROOT}//dataset/raw/django/
rm -rf ${PROJECT_ROOT}/dataset/tmp

# Preprocess
echo "--- Preprocess ---"
python ${PROJECT_ROOT}/main.py src.django.format_annotation \
	--annotation-file $PROJECT_ROOT/dataset/raw/django/all.anno \
	--destination $PROJECT_ROOT/dataset/raw/django/all_formatted.anno
python ${PROJECT_ROOT}/main.py src.django.split_dataset \
	--annotation-file $PROJECT_ROOT/dataset/raw/django/all_formatted.anno \
	--code-file $PROJECT_ROOT/dataset/raw/django/all.code \
	--destination $PROJECT_ROOT/dataset/django

ls ${PROJECT_ROOT}/dataset/django/train | cut -f 1 -d "." | sort | uniq | \
	xargs python ${PROJECT_ROOT}/main.py src.django.preprocess \
		--directory ${PROJECT_ROOT}dataset/django/train --ids
ls ${PROJECT_ROOT}/dataset/django/test  | cut -f 1 -d "." | sort | uniq | \
	xargs python ${PROJECT_ROOT}/main.py src.django.preprocess \
		--directory ${PROJECT_ROOT}dataset/django/test --ids
ls ${PROJECT_ROOT}/dataset/django/valid | cut -f 1 -d "." | sort | uniq | \
	xargs python ${PROJECT_ROOT}/main.py src.django.preprocess \
		--directory ${PROJECT_ROOT}dataset/django/valid --ids

ls ${PROJECT_ROOT}/dataset/django/train | cut -f 1 -d "." | sort | uniq | \
	xargs python ${PROJECT_ROOT}/main.py src.django.generate_action_sequence \
		--validate --directory ${PROJECT_ROOT}/dataset/django/train --ids 
ls ${PROJECT_ROOT}/dataset/django/test  | cut -f 1 -d "." | sort | uniq | \
	xargs python ${PROJECT_ROOT}/main.py src.django.generate_action_sequence \
		--validate --directory ${PROJECT_ROOT}/dataset/django/test --ids
ls ${PROJECT_ROOT}/dataset/django/valid | cut -f 1 -d "." | sort | uniq | \
	xargs python ${PROJECT_ROOT}/main.py src.django.generate_action_sequence \
	--validate --directory ${PROJECT_ROOT}/dataset/django/valid --ids
