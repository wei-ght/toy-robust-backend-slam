#!/usr/bin/env python3
"""
G2O 데이터셋의 EDGE_SE2 라인을 3번째 컬럼(destination node index) 기준으로 정렬하는 스크립트

Usage:
    python sort_g2o_edges.py input.g2o output.g2o
    
EDGE_SE2 format:
EDGE_SE2 source_id dest_id dx dy dtheta info(0,0) info(0,1) info(0,2) info(1,1) info(1,2) info(2,2)
"""

import sys
import os

def sort_g2o_file(input_file, output_file):
    """G2O 파일을 읽어서 EDGE_SE2를 3번째 컬럼 기준으로 정렬"""
    
    vertex_lines = []
    edge_lines = []
    other_lines = []
    
    print(f"Reading {input_file}...")
    
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
                    # EDGE_SE2 라인 파싱
                    parts = line.split()
                    if len(parts) >= 3:
                        source_id = int(parts[1])
                        dest_id = int(parts[2])
                        # (dest_id, source_id, original_line) 튜플로 저장하여 정렬
                        edge_lines.append((dest_id, source_id, line))
                    else:
                        print(f"Warning: Invalid EDGE_SE2 format at line {line_num}: {line}")
                        other_lines.append(line)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Error parsing line {line_num}: {line} - {e}")
                    other_lines.append(line)
            else:
                other_lines.append(line)
    
    print(f"Found {len(vertex_lines)} VERTEX_SE2 lines")
    print(f"Found {len(edge_lines)} EDGE_SE2 lines")
    print(f"Found {len(other_lines)} other lines")
    
    # EDGE_SE2를 destination node ID 기준으로 정렬
    # 동일한 destination이면 source ID로 2차 정렬
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
        
        # 각 destination으로 들어오는 에지 개수 통계
        dest_counts = {}
        for dest_id, _, _ in edge_lines:
            dest_counts[dest_id] = dest_counts.get(dest_id, 0) + 1
        
        max_incoming = max(dest_counts.values())
        max_dest_node = [k for k, v in dest_counts.items() if v == max_incoming][0]
        print(f"   - Max incoming edges: {max_incoming} (to node {max_dest_node})")

def main():
    if len(sys.argv) != 3:
        print("Usage: python sort_g2o_edges.py <input.g2o> <output.g2o>")
        print("\nExample:")
        print("  python sort_g2o_edges.py data/INTEL.g2o data/INTEL_sorted.g2o")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
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
        sort_g2o_file(input_file, output_file)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()