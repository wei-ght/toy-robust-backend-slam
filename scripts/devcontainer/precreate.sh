#!/bin/bash

# 폴더 생성 함수 정의
create_folder() {
    local folder_path="$1"
    if [ ! -d "$folder_path" ]; then
        mkdir -p "$folder_path"
        echo ".key 폴더가 존재 하지 않아 key폴더를 임시로 생성 하였습니다."
        echo "라이센스를 확인 해 주세요."
    fi
}

create_file() {
    local file_path="$1"
    if [ ! -e "$file_path" ]; then
        touch "$file_path"
    fi
}

# 경로 설정
folder_path="$HOME/.key"
folder_path2="$HOME/.ros"
folder_path3="$HOME/navifra_solution"
file_path="$HOME/.zsh_history2"
file_path2="$HOME/.gitconfig"
file_path3="$HOME/.git-credentials"

# 폴더 생성 함수 호출
create_folder "$folder_path"
create_folder "$folder_path2"
create_folder "$folder_path3"
create_file "$file_path"
create_file "$file_path2"
create_file "$file_path3"