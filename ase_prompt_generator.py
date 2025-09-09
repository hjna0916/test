import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import neighbor_list, natural_cutoffs

# --- 분석에 필요한 상수 정의 ---
FE_SYMBOL = 'Fe'
O_SYMBOL = 'O'
CRITICAL_DISTANCE = 2.5 
COORDINATION_CUTOFF = 3.0  

# --------------------------------------------------------------------------
# 분석 함수들
# --------------------------------------------------------------------------

def analyze_local_environment(atoms: Atoms, target_index: int) -> dict:
    distances = atoms.get_distances(target_index, range(len(atoms)), mic=True)
    min_dist_to_o = float('inf')
    coordination_count = 0
    for i, atom in enumerate(atoms):
        if atom.symbol == O_SYMBOL:
            dist = distances[i]
            if dist < min_dist_to_o:
                min_dist_to_o = dist
            if dist < COORDINATION_CUTOFF:
                coordination_count += 1
                
    return {"min_dist": min_dist_to_o, "coord_num": coordination_count}

def get_system_connectivity(atoms: Atoms) -> list:
    """시스템 전체의 원자 간 연결 정보(화학 결합)를 추출"""
    cutoffs = natural_cutoffs(atoms)
    indices_i, indices_j = neighbor_list('ij', atoms, cutoffs)
    bonds = [f"{atoms[i].symbol}{i}-{atoms[j].symbol}{j}" for i, j in zip(indices_i, indices_j) if i < j]
    return bonds

def create_rich_textual_representation(atoms: Atoms, current_step_number: int) -> str:
    """모든 ASE 분석을 종합하여 LLM을 위한 최종 리포트를 생성"""
    report_parts = []

    # 1. System Overview
    report_parts.append("## System Overview:")
    report_parts.append(f"- Total Atoms: {len(atoms)}")
    report_parts.append(f"- Chemical Formula: {atoms.get_chemical_formula(mode='hill')}\n")
    
    # 2. Fe 원자 분석 및 데이터 수집
    fe_indices = [atom.index for atom in atoms if atom.symbol == FE_SYMBOL]
    analysis_results = {idx: analyze_local_environment(atoms, idx) for idx in fe_indices}
    
    # 3. Descriptive Summary 생성
    summary_lines = [f"System contains {len(fe_indices)} Fe atoms."]
    for idx, result in analysis_results.items():
        status = "HIGH-RISK" if result['min_dist'] < CRITICAL_DISTANCE else "stable"
        line = f"Fe atom {idx} (Fe{idx}) is in a {status} state, coordinated by {result['coord_num']} water molecules. The nearest H2O distance is {result['min_dist']:.3f}Å."
        summary_lines.append(line)
        
    report_parts.append("## Descriptive Summary:")
    report_parts.append(" ".join(summary_lines) + "\n")
    
    # 4. Atom-specific Analysis 정리
    report_parts.append("## Atom-specific Analysis:")
    for idx, result in analysis_results.items():
        status = "HIGH-RISK (Critical distance rule triggered)" if result['min_dist'] < CRITICAL_DISTANCE else "Stable"
        report_parts.append(f"- Analysis for Fe atom {idx}:")
        report_parts.append(f"  - Status: {status}")
        report_parts.append(f"  - Nearest H2O (Oxygen) Distance: {result['min_dist']:.3f} Å")
        report_parts.append(f"  - H2O Coordination Count (within {COORDINATION_CUTOFF}Å): {result['coord_num']}")
    report_parts.append("")
    return "\n".join(report_parts)

def format_trajectory_data(trajectory_slice: list, step_offset: int) -> str:
    """
    ASE Atoms 객체 리스트를 LLM이 읽기 좋은 문자열로 변환
    """
    lines = []
    for i, atoms in enumerate(trajectory_slice):
        # 너무 길어지지 않게 각 스텝에서 Fe와 주변 O 몇 개만 표시 (예시)
        step_number = step_offset + i
        lines.append(f"Step {step_number}:")
        for atom in atoms:
            if atom.symbol == FE_SYMBOL:
                lines.append(f"  {atom.symbol}{atom.index}  {atom.x:.4f} {atom.y:.4f} {atom.z:.4f}")
    return "\n".join(lines)

def generate_full_prompt(current_atoms: Atoms, past_trajectory: list, current_step_num: int) -> str:
    """
    모든 분석 함수를 호출하여 최종 프롬프트를 조립하는 메인 함수입니다.
    """
    # 1. 변수 준비
    END_STEP = current_step_num
    final_report = create_rich_textual_representation(current_atoms, current_step_num)
    past_step_start_num = current_step_num - len(past_trajectory) + 1
    trajectory_data_string = format_trajectory_data(past_trajectory, past_step_start_num)

    # 2. 프롬프트 조립
    user_prompt = f"""You are a Computational Chemist specializing in Molecular Dynamics.
Your task is to predict the next atomic coordinates based on a detailed structural analysis and past trajectory.

-------------------------------------
**STRUCTURAL ANALYSIS REPORT (CURRENT STEP : {END_STEP})**
-------------------------------------
{final_report}
-------------------------------------
**PAST TRAJECTORY DATA (STPES {past_step_start_num} to {END_STEP - 1})**
-------------------------------------
{trajectory_data_string}
-------------------------------------

**YOUR TASK:**
Based on BOTH the analysis report and the past trajectory data, predict the atomic coordinates for the next step.
The atoms identified as 'HIGH-RISK' in the report should show significant displacement.
Provide the output in a single JSON format containing the coordinates for all atoms.
"""
    return user_prompt
