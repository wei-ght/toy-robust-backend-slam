#!/usr/bin/env python3
"""
G2O 데이터셋의 EDGE_SE2 라인을 3번째 컬럼(destination node index) 기준으로 정렬하고
필요시 edge direction을 swap하여 relative pose를 inverse 변환하는 스크립트

Usage:
    python sort_g2o_edges_with_inverse.py input.g2o output.g2o [--swap-direction]
    
EDGE_SE2 format:
EDGE_SE2 source_id dest_id dx dy dtheta info(0,0) info(0,1) info(0,2) info(1,1) info(1,2) info(2,2)

SE2 Inverse Transformation:
If edge A->B: T_AB = (dx, dy, dtheta)
Then edge B->A: T_BA = (-dx*cos(dtheta) - dy*sin(dtheta), dx*sin(dtheta) - dy*cos(dtheta), -dtheta)

Information Matrix Transformation:
For SE2 inverse, information matrix needs to be rotated by the inverse transformation
"""

import sys
import os
import math
import numpy as np

def se2_inverse(dx, dy, dtheta):
    """
    SE2 변환의 역변환 계산
    T_AB = (dx, dy, dtheta) -> T_BA = T_AB^(-1)
    
    Args:
        dx, dy, dtheta: SE2 transformation (x, y, theta)
        
    Returns:
        tuple: (inv_dx, inv_dy, inv_dtheta)
    """
    cos_theta = math.cos(dtheta)
    sin_theta = math.sin(dtheta)
    
    # SE2 inverse: T^(-1) = (-R^T * t, -theta)
    # where R is rotation matrix and t is translation
    inv_dx = -dx * cos_theta - dy * sin_theta
    inv_dy = dx * sin_theta - dy * cos_theta
    inv_dtheta = -dtheta
    
    return inv_dx, inv_dy, inv_dtheta

def transform_information_matrix(info_matrix, dtheta):
    """
    정보 행렬을 SE2 inverse 변환에 맞게 변환
    
    Args:
        info_matrix: 3x3 information matrix as [I00, I01, I02, I11, I12, I22]
        dtheta: rotation angle for transformation
        
    Returns:
        list: transformed information matrix in same format
    """
    # 정보 행렬을 3x3 형태로 복원
    I = np.zeros((3, 3))
    I[0, 0] = info_matrix[0]  # I00
    I[0, 1] = info_matrix[1]  # I01  
    I[0, 2] = info_matrix[2]  # I02
    I[1, 0] = info_matrix[1]  # I10 = I01 (symmetric)
    I[1, 1] = info_matrix[3]  # I11
    I[1, 2] = info_matrix[4]  # I12
    I[2, 0] = info_matrix[2]  # I20 = I02 (symmetric)
    I[2, 1] = info_matrix[4]  # I21 = I12 (symmetric)
    I[2, 2] = info_matrix[5]  # I22
    
    # SE2 inverse에 대한 변환 행렬 (Jacobian)
    cos_theta = math.cos(dtheta)
    sin_theta = math.sin(dtheta)
    
    # J = [[cos(θ), sin(θ), 0],
    #      [-sin(θ), cos(θ), 0],
    #      [0,       0,      -1]]
    J = np.array([
        [cos_theta, sin_theta, 0],
        [-sin_theta, cos_theta, 0], 
        [0, 0, -1]
    ])
    
    # 변환된 정보 행렬: I' = J^T * I * J
    I_transformed = J.T @ I @ J
    
    # 다시 compressed format으로 변환
    return [
        I_transformed[0, 0],  # I00
        I_transformed[0, 1],  # I01
        I_transformed[0, 2],  # I02
        I_transformed[1, 1],  # I11
        I_transformed[1, 2],  # I12
        I_transformed[2, 2]   # I22
    ]

def parse_edge_line(line):
    """EDGE_SE2 라인을 파싱하여 구성요소들을 반환"""
    parts = line.split()
    if len(parts) < 12:
        raise ValueError(f"Invalid EDGE_SE2 format: {line}")
    
    source_id = int(parts[1])
    dest_id = int(parts[2])
    dx = float(parts[3])
    dy = float(parts[4])
    dtheta = float(parts[5])
    
    # Information matrix (상삼각 형태)
    info_matrix = [float(parts[i]) for i in range(6, 12)]
    
    return source_id, dest_id, dx, dy, dtheta, info_matrix

def format_edge_line(source_id, dest_id, dx, dy, dtheta, info_matrix):
    """구성요소들을 EDGE_SE2 라인으로 포맷"""
    return (f"EDGE_SE2 {source_id} {dest_id} {dx:.6f} {dy:.6f} {dtheta:.6f} "
            f"{info_matrix[0]:.6f} {info_matrix[1]:.6f} {info_matrix[2]:.6f} "
            f"{info_matrix[3]:.6f} {info_matrix[4]:.6f} {info_matrix[5]:.6f}")

def sort_g2o_file(input_file, output_file, swap_direction=False):
    """G2O 파일을 읽어서 EDGE_SE2를 3번째 컬럼 기준으로 정렬"""
    
    vertex_lines = []
    edge_lines = []
    other_lines = []
    swapped_count = 0
    
    print(f"Reading {input_file}...")
    print(f"Swap direction mode: {'ON' if swap_direction else 'OFF'}")
    
    # 파일 읽기
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 빈 줄 건너뛰기
                continue
                
            if line.startswith('VERTEX_SE2'):
                vertex_lines.append(line)
            elif line.startswith('EDGE_SE2'):
                try:
                    source_id, dest_id, dx, dy, dtheta, info_matrix = parse_edge_line(line)
                    
                    # 방향 스왑 옵션이 켜져있고 source > dest인 경우 inverse 적용
                    if swap_direction and source_id > dest_id:
                        if swapped_count < 5:  # 처음 5개만 출력
                            print(f"  Swapping edge {source_id}->{dest_id} to {dest_id}->{source_id}")
                        elif swapped_count == 5:
                            print(f"  ... (more edges swapped)")
                        swapped_count += 1
                        
                        # SE2 inverse 변환
                        inv_dx, inv_dy, inv_dtheta = se2_inverse(dx, dy, dtheta)
                        
                        # 정보 행렬 변환
                        inv_info_matrix = transform_information_matrix(info_matrix, dtheta)
                        
                        # source와 dest 교체
                        edge_lines.append((dest_id, source_id, format_edge_line(
                            dest_id, source_id, inv_dx, inv_dy, inv_dtheta, inv_info_matrix)))
                    else:
                        # 원본 에지 그대로 사용
                        edge_lines.append((dest_id, source_id, line))
                        
                except (ValueError, IndexError) as e:
                    print(f"Warning: Error parsing line {line_num}: {line} - {e}")
                    other_lines.append(line)
            else:
                other_lines.append(line)
    
    print(f"Found {len(vertex_lines)} VERTEX_SE2 lines")
    print(f"Found {len(edge_lines)} EDGE_SE2 lines")
    if swap_direction:
        print(f"Swapped {swapped_count} edges (source > dest)")
    print(f"Found {len(other_lines)} other lines")
    
    # EDGE_SE2를 destination node ID 기준으로 정렬
    edge_lines.sort(key=lambda x: (x[0], x[1]))
    
    print(f"Writing sorted data to {output_file}...")
    
    # 정렬된 결과를 파일에 쓰기
    with open(output_file, 'w') as f:
        # 1. VERTEX_SE2 라인들 먼저 출력
        for vertex_line in vertex_lines:
            f.write(vertex_line + '\n')
        
        # 2. 빈 줄 추가 (가독성을 위해)
        if vertex_lines and edge_lines:
            f.write('\n')
        
        # 3. 정렬된 EDGE_SE2 라인들 출력
        for dest_id, source_id, edge_line in edge_lines:
            f.write(edge_line + '\n')
        
        # 4. 기타 라인들 출력
        if other_lines:
            f.write('\n')
            for other_line in other_lines:
                f.write(other_line + '\n')
    
    print(f"✅ Sorting completed!")
    print(f"📊 Statistics:")
    if edge_lines:
        min_dest = min(edge_lines, key=lambda x: x[0])[0]
        max_dest = max(edge_lines, key=lambda x: x[0])[0]
        print(f"   - Destination node range: {min_dest} ~ {max_dest}")

def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python sort_g2o_edges_with_inverse.py <input.g2o> <output.g2o> [--swap-direction]")
        print("\nOptions:")
        print("  --swap-direction  : Swap edge direction and apply SE2 inverse when source_id > dest_id")
        print("\nExamples:")
        print("  python sort_g2o_edges_with_inverse.py data/INTEL.g2o data/INTEL_sorted.g2o")
        print("  python sort_g2o_edges_with_inverse.py data/INTEL.g2o data/INTEL_inverse.g2o --swap-direction")
        print("\nInverse Transformation Example:")
        print("  Original: EDGE_SE2 166 19 -2.459689 0.241111 0.252800 ...")
        print("  Becomes:  EDGE_SE2 19 166 [inverse_transform] ...")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    swap_direction = len(sys.argv) == 4 and sys.argv[3] == '--swap-direction'
    
    # 입력 파일 존재 확인
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' does not exist!")
        sys.exit(1)
    
    # 출력 디렉토리 생성
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created directory: {output_dir}")
    
    try:
        sort_g2o_file(input_file, output_file, swap_direction)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()