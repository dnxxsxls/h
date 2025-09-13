#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합된 만능 로컬 극값 탐지 시스템
모든 탐지 방법을 하나로 통합하여 데이터 특성에 따라 자동으로 최적의 방법을 선택합니다.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import os


class UnifiedExtremaDetector:
    """
    통합된 만능 로컬 극값 탐지기
    
    데이터의 특성을 자동으로 분석하여 최적의 탐지 방법을 선택하고,
    여러 방법의 결과를 종합하여 가장 정확한 극값을 찾습니다.
    """
    
    def __init__(self):
        self.data_stats = {}
        self.detection_methods = {
            'simple': self._detect_simple,
            'window': self._detect_window,
            'slope': self._detect_slope,
            'alternating': self._detect_alternating,
            'enhanced': self._detect_enhanced,
            'strict': self._detect_strict
        }
    
    def analyze_data_characteristics(self, data: List[float]) -> Dict:
        """
        데이터의 특성을 분석하여 최적의 탐지 방법을 결정합니다.
        
        Args:
            data: 분석할 데이터
            
        Returns:
            데이터 특성 정보 딕셔너리
        """
        if not data:
            return {}
        
        data_array = np.array(data)
        
        # 기본 통계
        min_val = np.min(data_array)
        max_val = np.max(data_array)
        mean_val = np.mean(data_array)
        std_val = np.std(data_array)
        range_val = max_val - min_val
        
        # 데이터 길이
        data_length = len(data)
        
        # 노이즈 레벨 추정 (연속된 값들의 차이의 표준편차)
        diffs = np.diff(data_array)
        noise_level = np.std(diffs)
        
        # 데이터의 변동성 (전체 범위 대비 표준편차 비율)
        variability = std_val / range_val if range_val > 0 else 0
        
        # 극값 밀도 추정 (간단한 방법으로 예비 극값 개수 계산)
        simple_extrema = self._detect_simple(data, threshold=noise_level * 0.1)
        estimated_extrema_count = len(simple_extrema[0]) + len(simple_extrema[1])
        extrema_density = estimated_extrema_count / data_length if data_length > 0 else 0
        
        # 데이터 패턴 분석
        is_oscillatory = self._is_oscillatory_pattern(data_array)
        has_plateaus = self._has_plateaus(data_array, noise_level)
        
        characteristics = {
            'length': data_length,
            'min_val': min_val,
            'max_val': max_val,
            'mean_val': mean_val,
            'std_val': std_val,
            'range_val': range_val,
            'noise_level': noise_level,
            'variability': variability,
            'extrema_density': extrema_density,
            'is_oscillatory': is_oscillatory,
            'has_plateaus': has_plateaus,
            'estimated_extrema_count': estimated_extrema_count
        }
        
        self.data_stats = characteristics
        return characteristics
    
    def _is_oscillatory_pattern(self, data: np.ndarray) -> bool:
        """데이터가 진동 패턴을 보이는지 확인합니다."""
        if len(data) < 10:
            return False
        
        # 기울기의 부호 변화 빈도 계산
        diffs = np.diff(data)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        change_rate = sign_changes / len(diffs) if len(diffs) > 0 else 0
        
        return change_rate > 0.3  # 30% 이상의 부호 변화가 있으면 진동 패턴
    
    def _has_plateaus(self, data: np.ndarray, noise_level: float) -> bool:
        """데이터에 평평한 구간(plateau)이 있는지 확인합니다."""
        if len(data) < 5:
            return False
        
        # 연속된 값들이 비슷한 구간이 있는지 확인
        diffs = np.abs(np.diff(data))
        plateau_threshold = noise_level * 0.5
        
        consecutive_similar = 0
        max_consecutive = 0
        
        for diff in diffs:
            if diff < plateau_threshold:
                consecutive_similar += 1
                max_consecutive = max(max_consecutive, consecutive_similar)
            else:
                consecutive_similar = 0
        
        return max_consecutive >= 5  # 5개 이상 연속으로 비슷한 값이 있으면 plateau
    
    def select_optimal_method(self, characteristics: Dict) -> str:
        """
        데이터 특성에 따라 최적의 탐지 방법을 선택합니다.
        
        Args:
            characteristics: 데이터 특성 정보
            
        Returns:
            선택된 탐지 방법 이름
        """
        length = characteristics.get('length', 0)
        variability = characteristics.get('variability', 0)
        extrema_density = characteristics.get('extrema_density', 0)
        is_oscillatory = characteristics.get('is_oscillatory', False)
        has_plateaus = characteristics.get('has_plateaus', False)
        noise_level = characteristics.get('noise_level', 0)
        
        # 데이터 길이가 매우 짧은 경우
        if length < 10:
            return 'simple'
        
        # 진동 패턴이 강한 경우
        if is_oscillatory and extrema_density > 0.1:
            return 'alternating'
        
        # 평평한 구간이 많은 경우
        if has_plateaus:
            return 'window'
        
        # 노이즈가 많은 경우
        if noise_level > characteristics.get('std_val', 1) * 0.1:
            return 'enhanced'
        
        # 극값이 매우 많은 경우 (노이즈가 많을 가능성)
        if extrema_density > 0.2:
            return 'strict'
        
        # 변동성이 낮은 경우
        if variability < 0.1:
            return 'slope'
        
        # 기본적으로는 향상된 방법 사용
        return 'enhanced'
    
    def detect_extrema(self, data: List[float], method: Optional[str] = None, 
                      **kwargs) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """
        통합된 극값 탐지 메인 함수
        
        Args:
            data: 분석할 데이터
            method: 사용할 탐지 방법 (None이면 자동 선택)
            **kwargs: 각 방법별 추가 파라미터
            
        Returns:
            (minima, maxima) 튜플
        """
        if not data or len(data) < 3:
            return [], []
        
        # 데이터 특성 분석
        characteristics = self.analyze_data_characteristics(data)
        
        # 탐지 방법 선택
        if method is None:
            method = self.select_optimal_method(characteristics)
        
        # 선택된 방법으로 탐지 실행
        if method in self.detection_methods:
            minima, maxima = self.detection_methods[method](data, **kwargs)
        else:
            # 기본 방법 사용
            minima, maxima = self._detect_enhanced(data, **kwargs)
        
        # 결과 후처리 및 검증
        minima, maxima = self._post_process_results(minima, maxima, data, characteristics)
        
        return minima, maxima
    
    def _detect_simple(self, data: List[float], threshold: float = 0.0001) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """단순한 방법으로 로컬 극값을 찾습니다."""
        if len(data) < 3:
            return [], []
        
        minima = []
        maxima = []
        
        for i in range(1, len(data) - 1):
            prev_val = data[i-1]
            curr_val = data[i]
            next_val = data[i+1]
            
            # 최대값: 이전 값보다 크고, 다음 값보다 크거나 같음
            if curr_val > prev_val + threshold and curr_val >= next_val - threshold:
                maxima.append((i, curr_val))
            # 최소값: 이전 값보다 작고, 다음 값보다 작거나 같음
            elif curr_val < prev_val - threshold and curr_val <= next_val + threshold:
                minima.append((i, curr_val))
        
        return minima, maxima
    
    def _detect_window(self, data: List[float], window_size: int = 3, threshold: float = 0.0001, **kwargs) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """윈도우 기반 방법으로 로컬 극값을 찾습니다."""
        if len(data) < window_size + 2:
            return [], []
        
        minima = []
        maxima = []
        
        for i in range(window_size, len(data) - window_size):
            current_value = data[i]
            
            left_window = data[i-window_size:i]
            right_window = data[i+1:i+window_size+1]
            
            # 로컬 최대값인지 확인
            is_local_max = all(current_value >= val for val in left_window + right_window)
            
            # 로컬 최소값인지 확인
            is_local_min = all(current_value <= val for val in left_window + right_window)
            
            if is_local_max:
                maxima.append((i, current_value))
            elif is_local_min:
                minima.append((i, current_value))
        
        return minima, maxima
    
    def _detect_slope(self, data: List[float], threshold: float = 0.0001, window_size: int = 3, **kwargs) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """기울기 변화를 이용하여 로컬 극값을 찾습니다."""
        if len(data) < 4:
            return [], []
        
        # 기울기 계산
        slopes = []
        for i in range(len(data) - 1):
            slope = data[i+1] - data[i]
            slopes.append(slope)
        
        minima = []
        maxima = []
        
        # 기울기의 부호 변화를 찾아서 극값 탐지
        for i in range(1, len(slopes) - 1):
            prev_slope = slopes[i-1]
            curr_slope = slopes[i]
            
            if abs(curr_slope) < threshold:
                continue
            
            # 양에서 음으로 변하면 최대값
            if prev_slope > threshold and curr_slope < -threshold:
                maxima.append((i, data[i]))
            # 음에서 양으로 변하면 최소값
            elif prev_slope < -threshold and curr_slope > threshold:
                minima.append((i, data[i]))
        
        return minima, maxima
    
    def _detect_alternating(self, data: List[float], threshold: float = 0.0001, window_size: int = 3, **kwargs) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """교차 패턴을 보장하는 로컬 극값 탐지 방법"""
        if len(data) < 5:
            return [], []
        
        # 먼저 모든 극값을 찾기
        all_maxima = []
        all_minima = []
        
        for i in range(2, len(data) - 2):
            prev_val = data[i-1]
            curr_val = data[i]
            next_val = data[i+1]
            
            # 최대값 확인 (더 엄격한 조건)
            if (curr_val > prev_val + threshold and 
                curr_val > next_val + threshold and
                curr_val > data[i-2] + threshold and
                curr_val > data[i+2] + threshold):
                all_maxima.append((i, curr_val))
            
            # 최소값 확인 (더 엄격한 조건)
            elif (curr_val < prev_val - threshold and 
                  curr_val < next_val - threshold and
                  curr_val < data[i-2] - threshold and
                  curr_val < data[i+2] - threshold):
                all_minima.append((i, curr_val))
        
        # 교차 패턴으로 정렬
        extrema = []
        
        # 모든 극값을 인덱스 순으로 정렬
        all_extrema = []
        for idx, val in all_maxima:
            all_extrema.append(('max', idx, val))
        for idx, val in all_minima:
            all_extrema.append(('min', idx, val))
        
        all_extrema.sort(key=lambda x: x[1])  # 인덱스 순으로 정렬
        
        # 교차 패턴 적용
        for typ, idx, val in all_extrema:
            if not extrema:
                # 첫 번째 극값은 무조건 추가
                extrema.append((typ, idx, val))
            else:
                last_type = extrema[-1][0]
                
                # 교차 패턴 확인
                if typ != last_type:
                    # 충분한 변화가 있는지 확인
                    last_val = extrema[-1][2]
                    if typ == 'max' and val > last_val + threshold:
                        extrema.append((typ, idx, val))
                    elif typ == 'min' and val < last_val - threshold:
                        extrema.append((typ, idx, val))
        
        # 최대값과 최소값으로 분리
        maxima = [(idx, val) for typ, idx, val in extrema if typ == 'max']
        minima = [(idx, val) for typ, idx, val in extrema if typ == 'min']
        
        return minima, maxima
    
    def _detect_enhanced(self, data: List[float], threshold: float = 0.0001, window_size: int = 3, **kwargs) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """향상된 교차 패턴 로컬 극값 탐지 방법"""
        if len(data) < 7:
            return [], []
        
        # 데이터의 전체 범위 파악
        min_val = min(data)
        max_val = max(data)
        data_range = max_val - min_val
        
        # 최소값 임계값: 0.4 이하 또는 전체 범위의 30% 이하
        min_threshold = min(0.4, min_val + data_range * 0.3)
        
        # 1단계: 모든 잠재적 극값 찾기
        potential_maxima = []
        potential_minima = []
        
        for i in range(3, len(data) - 3):
            curr_val = data[i]
            
            # 최대값 후보: 주변 3개씩 모두보다 큼
            is_max = True
            for j in range(i-3, i+4):
                if j != i and data[j] >= curr_val:
                    is_max = False
                    break
            
            if is_max:
                potential_maxima.append((i, curr_val))
            
            # 최소값 후보: 주변 3개씩 모두보다 작고, 임계값 이하
            is_min = True
            if curr_val > min_threshold:
                is_min = False
            else:
                for j in range(i-3, i+4):
                    if j != i and data[j] <= curr_val:
                        is_min = False
                        break
            
            if is_min:
                potential_minima.append((i, curr_val))
        
        # 2단계: 교차 패턴으로 정렬
        all_extrema = []
        for idx, val in potential_maxima:
            all_extrema.append(('max', idx, val))
        for idx, val in potential_minima:
            all_extrema.append(('min', idx, val))
        
        # 인덱스 순으로 정렬
        all_extrema.sort(key=lambda x: x[1])
        
        # 3단계: 교차 패턴 적용 및 충분한 변화 확인
        final_extrema = []
        
        for typ, idx, val in all_extrema:
            if not final_extrema:
                # 첫 번째 극값은 무조건 추가
                final_extrema.append((typ, idx, val))
            else:
                last_type, last_idx, last_val = final_extrema[-1]
                
                # 교차 패턴 확인
                if typ != last_type:
                    # 충분한 거리와 변화가 있는지 확인
                    distance = idx - last_idx
                    value_change = abs(val - last_val)
                    
                    # 최소 거리와 변화량 조건
                    if distance >= 2 and value_change >= threshold:
                        # 추가 조건: 최소값은 임계값 이하이고 이전 최대값보다 충분히 작아야 함
                        if typ == 'min' and val <= min_threshold and val < last_val - threshold * 2:
                            final_extrema.append((typ, idx, val))
                        # 추가 조건: 최대값은 이전 최소값보다 충분히 커야 함
                        elif typ == 'max' and val > last_val + threshold * 2:
                            final_extrema.append((typ, idx, val))
        
        # 4단계: 최종 결과 분리
        maxima = [(idx, val) for typ, idx, val in final_extrema if typ == 'max']
        minima = [(idx, val) for typ, idx, val in final_extrema if typ == 'min']
        
        return minima, maxima
    
    def _detect_strict(self, data: List[float], threshold: float = 0.0001, window_size: int = 3, **kwargs) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """엄격한 최소값 조건을 적용한 로컬 극값 탐지 방법"""
        if len(data) < 7:
            return [], []
        
        # 데이터의 전체 범위 파악
        min_val = min(data)
        max_val = max(data)
        data_range = max_val - min_val
        
        # 최소값 임계값: 0.4 고정
        min_threshold = 0.4
        
        # 1단계: 모든 잠재적 극값 찾기
        potential_maxima = []
        potential_minima = []
        
        for i in range(3, len(data) - 3):
            curr_val = data[i]
            
            # 최대값 후보: 주변 3개씩 모두보다 큼
            is_max = True
            for j in range(i-3, i+4):
                if j != i and data[j] >= curr_val:
                    is_max = False
                    break
            
            if is_max:
                potential_maxima.append((i, curr_val))
            
            # 최소값 후보: 주변 3개씩 모두보다 작고, 반드시 0.4 이하
            is_min = True
            if curr_val > min_threshold:
                is_min = False  # 0.4보다 크면 절대 최소값이 될 수 없음
            else:
                for j in range(i-3, i+4):
                    if j != i and data[j] <= curr_val:
                        is_min = False
                        break
            
            if is_min:
                potential_minima.append((i, curr_val))
        
        # 2단계: 교차 패턴으로 정렬
        all_extrema = []
        for idx, val in potential_maxima:
            all_extrema.append(('max', idx, val))
        for idx, val in potential_minima:
            all_extrema.append(('min', idx, val))
        
        # 인덱스 순으로 정렬
        all_extrema.sort(key=lambda x: x[1])
        
        # 3단계: 교차 패턴 적용 및 엄격한 조건 확인
        final_extrema = []
        
        for typ, idx, val in all_extrema:
            if not final_extrema:
                # 첫 번째 극값은 조건에 맞으면 추가
                if typ == 'max' or (typ == 'min' and val <= min_threshold):
                    final_extrema.append((typ, idx, val))
            else:
                last_type, last_idx, last_val = final_extrema[-1]
                
                # 교차 패턴 확인
                if typ != last_type:
                    # 충분한 거리와 변화가 있는지 확인
                    distance = idx - last_idx
                    value_change = abs(val - last_val)
                    
                    # 최소 거리와 변화량 조건
                    if distance >= 3 and value_change >= threshold:
                        # 엄격한 조건: 최소값은 반드시 0.4 이하이고 이전 최대값보다 충분히 작아야 함
                        if typ == 'min' and val <= min_threshold and val < last_val - threshold * 3:
                            final_extrema.append((typ, idx, val))
                        # 최대값은 이전 최소값보다 충분히 커야 함
                        elif typ == 'max' and val > last_val + threshold * 3:
                            final_extrema.append((typ, idx, val))
        
        # 4단계: 최종 결과 분리
        maxima = [(idx, val) for typ, idx, val in final_extrema if typ == 'max']
        minima = [(idx, val) for typ, idx, val in final_extrema if typ == 'min']
        
        return minima, maxima
    
    def _post_process_results(self, minima: List[Tuple[int, float]], maxima: List[Tuple[int, float]], 
                            data: List[float], characteristics: Dict) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """
        탐지 결과를 후처리하여 품질을 향상시킵니다.
        
        Args:
            minima: 탐지된 최소값들
            maxima: 탐지된 최대값들
            data: 원본 데이터
            characteristics: 데이터 특성
            
        Returns:
            후처리된 (minima, maxima) 튜플
        """
        # 중복 제거 (같은 인덱스의 극값이 중복된 경우)
        minima_dict = {idx: val for idx, val in minima}
        maxima_dict = {idx: val for idx, val in maxima}
        
        # 인덱스 순으로 정렬
        minima = sorted(minima_dict.items())
        maxima = sorted(maxima_dict.items())
        
        # 너무 가까운 극값들 제거 (최소 거리 조건)
        min_distance = max(2, len(data) // 100)  # 데이터 길이의 1% 또는 최소 2
        
        minima = self._remove_close_extrema(minima, min_distance)
        maxima = self._remove_close_extrema(maxima, min_distance)
        
        # 극값의 품질 검증
        noise_level = characteristics.get('noise_level', 0)
        if noise_level > 0:
            minima = self._filter_by_quality(minima, data, noise_level, is_minimum=True)
            maxima = self._filter_by_quality(maxima, data, noise_level, is_minimum=False)
        
        return minima, maxima
    
    def _remove_close_extrema(self, extrema: List[Tuple[int, float]], min_distance: int) -> List[Tuple[int, float]]:
        """너무 가까운 극값들을 제거합니다."""
        if not extrema:
            return []
        
        filtered = [extrema[0]]  # 첫 번째는 항상 유지
        
        for i in range(1, len(extrema)):
            current_idx = extrema[i][0]
            last_kept_idx = filtered[-1][0]
            
            if current_idx - last_kept_idx >= min_distance:
                filtered.append(extrema[i])
        
        return filtered
    
    def _filter_by_quality(self, extrema: List[Tuple[int, float]], data: List[float], 
                          noise_level: float, is_minimum: bool) -> List[Tuple[int, float]]:
        """극값의 품질을 검증하여 필터링합니다."""
        if not extrema:
            return []
        
        filtered = []
        
        for idx, val in extrema:
            # 주변 데이터와의 차이 계산
            window_size = min(3, len(data) // 10)
            start_idx = max(0, idx - window_size)
            end_idx = min(len(data), idx + window_size + 1)
            
            local_data = data[start_idx:end_idx]
            if not local_data:
                continue
            
            # 극값이 주변 데이터와 충분히 다른지 확인
            if is_minimum:
                min_local = min(local_data)
                if val <= min_local + noise_level * 2:  # 최소값인 경우
                    filtered.append((idx, val))
            else:
                max_local = max(local_data)
                if val >= max_local - noise_level * 2:  # 최대값인 경우
                    filtered.append((idx, val))
        
        return filtered
    
    def get_detection_info(self) -> Dict:
        """현재 탐지 설정 정보를 반환합니다."""
        return {
            'data_stats': self.data_stats,
            'available_methods': list(self.detection_methods.keys()),
            'selected_method': getattr(self, 'last_method', 'auto')
        }


def read_data_file(file_path: str) -> List[float]:
    """
    데이터 파일을 읽어서 float 리스트로 반환합니다.
    
    Args:
        file_path: 데이터 파일 경로
        
    Returns:
        float 값들의 리스트
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    try:
                        value = float(line)
                        data.append(value)
                    except ValueError:
                        print(f"경고: '{line}'는 유효한 숫자가 아닙니다. 건너뜁니다.")
        print(f"{file_path}에서 {len(data)}개의 데이터를 읽었습니다.")
        return data
    except FileNotFoundError:
        print(f"오류: 파일 '{file_path}'을 찾을 수 없습니다.")
        return []
    except Exception as e:
        print(f"오류: 파일을 읽는 중 문제가 발생했습니다: {e}")
        return []


def main():
    """메인 함수: 통합된 탐지 시스템을 테스트합니다."""
    print("통합된 만능 로컬 극값 탐지 시스템을 시작합니다...\n")
    
    # 테스트할 파일들
    test_files = ['cube 1.txt', 'cube 2.txt', 'cube 3.txt', 'cube 4.txt']
    
    # 탐지기 생성
    detector = UnifiedExtremaDetector()
    
    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"파일을 찾을 수 없습니다: {file_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"파일 분석: {file_path}")
        print(f"{'='*60}")
        
        # 데이터 읽기
        data = read_data_file(file_path)
        if not data:
            continue
        
        # 자동 탐지 실행
        print("\n자동 탐지 실행 중...")
        minima, maxima = detector.detect_extrema(data)
        
        # 탐지 정보 출력
        info = detector.get_detection_info()
        print(f"\n데이터 특성:")
        print(f"  - 데이터 길이: {info['data_stats'].get('length', 0)}")
        print(f"  - 변동성: {info['data_stats'].get('variability', 0):.4f}")
        print(f"  - 노이즈 레벨: {info['data_stats'].get('noise_level', 0):.6f}")
        print(f"  - 극값 밀도: {info['data_stats'].get('extrema_density', 0):.4f}")
        print(f"  - 진동 패턴: {'예' if info['data_stats'].get('is_oscillatory', False) else '아니오'}")
        print(f"  - 평평한 구간: {'예' if info['data_stats'].get('has_plateaus', False) else '아니오'}")
        
        # 결과 출력
        print(f"\n탐지 결과:")
        print(f"  - 로컬 최대값: {len(maxima)}개")
        print(f"  - 로컬 최소값: {len(minima)}개")
        
        if maxima:
            print(f"\n최대값들 (처음 10개):")
            for i, (idx, val) in enumerate(maxima[:10]):
                print(f"  {i+1:2d}. 인덱스: {idx:4d}, 값: {val:12.8f}")
        
        if minima:
            print(f"\n최소값들 (처음 10개):")
            for i, (idx, val) in enumerate(minima[:10]):
                print(f"  {i+1:2d}. 인덱스: {idx:4d}, 값: {val:12.8f}")
        
        # 결과 저장
        output_file = f"{file_path.replace('.txt', '')}_unified_results.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== {file_path} 통합 탐지 결과 ===\n")
            f.write(f"탐지 방법: 자동 선택\n")
            f.write(f"로컬 최대값 개수: {len(maxima)}\n")
            f.write(f"로컬 최소값 개수: {len(minima)}\n\n")
            
            if maxima:
                f.write("로컬 최대값들:\n")
                for i, (idx, val) in enumerate(maxima):
                    f.write(f"{i+1:3d}. 인덱스: {idx:4d}, 값: {val:12.8f}\n")
            
            if minima:
                f.write("\n로컬 최소값들:\n")
                for i, (idx, val) in enumerate(minima):
                    f.write(f"{i+1:3d}. 인덱스: {idx:4d}, 값: {val:12.8f}\n")
        
        print(f"\n결과가 {output_file}에 저장되었습니다.")
    
    print(f"\n{'='*60}")
    print("통합 탐지 시스템 테스트가 완료되었습니다!")
    print("모든 파일에 대해 자동으로 최적의 탐지 방법을 선택하여 분석했습니다.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
