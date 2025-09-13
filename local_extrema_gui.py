#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
로컬 극값 탐지 GUI 프로그램
사용자가 파일을 업로드하고 파라미터를 설정하여 로컬 최소값/최대값을 찾을 수 있습니다.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.font_manager as fm
import numpy as np
from unified_extrema_detector import UnifiedExtremaDetector


class LocalExtremaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("통합된 만능 로컬 극값 탐지 프로그램")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # 데이터 저장 변수
        self.data = []
        self.x_data = []
        self.y_data = []
        self.current_file_path = ""
        self.results = {}
        self.figure = None
        self.canvas = None
        
        # 통합 탐지기 초기화
        self.detector = UnifiedExtremaDetector()
        
        # 인터랙티브 선택 변수
        self.selected_maxima = []  # 사용자가 선택한 최대값들
        self.selected_minima = []  # 사용자가 선택한 최소값들
        self.interactive_mode = False  # 인터랙티브 모드 상태
        
        # 차이값 계산 결과 저장
        self.difference_results = []  # 최대값-최소값 차이값들
        
        # 한글 폰트 설정
        self.setup_korean_font()
        
        # GUI 구성 요소 생성
        self.create_widgets()
    
    def setup_korean_font(self):
        """한글 폰트를 설정합니다."""
        try:
            # Windows에서 사용 가능한 한글 폰트 찾기
            font_candidates = [
                'Malgun Gothic',  # Windows 10/11
                'AppleGothic',    # macOS
                'Noto Sans CJK KR',  # Linux
                'DejaVu Sans'
            ]
            
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            
            for font in font_candidates:
                if font in available_fonts:
                    plt.rcParams['font.family'] = font
                    break
            else:
                # 한글 폰트를 찾지 못한 경우 기본 폰트 사용
                plt.rcParams['font.family'] = 'DejaVu Sans'
                
        except Exception as e:
            print(f"폰트 설정 중 오류: {e}")
            plt.rcParams['font.family'] = 'DejaVu Sans'
        
    def create_widgets(self):
        """GUI 위젯들을 생성합니다."""
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 왼쪽 프레임 (컨트롤)
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # 오른쪽 프레임 (그래프)
        right_frame = ttk.LabelFrame(main_frame, text="데이터 그래프", padding="5")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 제목
        title_label = ttk.Label(left_frame, text="통합된 만능 로컬 극값 탐지 프로그램", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 파일 선택 섹션
        file_frame = ttk.LabelFrame(left_frame, text="1. 데이터 파일 선택", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.file_label = ttk.Label(file_frame, text="파일을 선택하세요")
        self.file_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        ttk.Button(file_frame, text="파일 선택", 
                  command=self.select_file).grid(row=0, column=1)
        
        # 파일 정보 표시
        self.file_info_label = ttk.Label(file_frame, text="", foreground='blue')
        self.file_info_label.grid(row=1, column=0, columnspan=2, pady=(5, 0))
        
        # 파라미터 설정 섹션
        param_frame = ttk.LabelFrame(left_frame, text="2. 탐지 파라미터 설정", padding="10")
        param_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 인터랙티브 모드 선택
        ttk.Label(param_frame, text="탐지 모드:").grid(row=0, column=0, sticky=tk.W)
        self.mode_var = tk.StringVar(value="auto")
        mode_frame = ttk.Frame(param_frame)
        mode_frame.grid(row=0, column=1, columnspan=2, sticky=tk.W, padx=(10, 0))
        
        ttk.Radiobutton(mode_frame, text="자동 탐지", variable=self.mode_var, 
                       value="auto", command=self.toggle_mode).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(mode_frame, text="수동 선택", variable=self.mode_var, 
                       value="manual", command=self.toggle_mode).pack(side=tk.LEFT)
        
        # 자동 모드 파라미터 (기본값)
        self.auto_params_frame = ttk.Frame(param_frame)
        self.auto_params_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # 최대값 개수 제한
        ttk.Label(self.auto_params_frame, text="최대값 개수 제한:").grid(row=0, column=0, sticky=tk.W)
        self.max_count_var = tk.StringVar(value="10")
        max_count_entry = ttk.Entry(self.auto_params_frame, textvariable=self.max_count_var, width=10)
        max_count_entry.grid(row=0, column=1, padx=(10, 0))
        ttk.Label(self.auto_params_frame, text="개 (0이면 제한 없음)").grid(row=0, column=2, sticky=tk.W, padx=(5, 0))
        
        # 최소값 개수 제한
        ttk.Label(self.auto_params_frame, text="최소값 개수 제한:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.min_count_var = tk.StringVar(value="10")
        min_count_entry = ttk.Entry(self.auto_params_frame, textvariable=self.min_count_var, width=10)
        min_count_entry.grid(row=1, column=1, padx=(10, 0), pady=(5, 0))
        ttk.Label(self.auto_params_frame, text="개 (0이면 제한 없음)").grid(row=1, column=2, sticky=tk.W, padx=(5, 0), pady=(5, 0))
        
        # 수동 모드 파라미터 (숨김)
        self.manual_params_frame = ttk.Frame(param_frame)
        self.manual_params_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # 수동 선택 안내
        manual_info = ttk.Label(self.manual_params_frame, 
                               text="그래프에서 좌클릭: 최대값 선택, 우클릭: 최소값 선택", 
                               font=('Arial', 9), foreground='blue')
        manual_info.grid(row=0, column=0, columnspan=3, pady=(5, 0))
        
        # 선택된 점 개수 표시
        self.selection_info = ttk.Label(self.manual_params_frame, 
                                       text="선택된 최대값: 0개, 최소값: 0개", 
                                       font=('Arial', 9), foreground='green')
        self.selection_info.grid(row=1, column=0, columnspan=3, pady=(5, 0))
        
        # 선택 초기화 버튼
        ttk.Button(self.manual_params_frame, text="선택 초기화", 
                  command=self.clear_selections).grid(row=2, column=0, pady=(5, 0))
        
        # 수동 모드 프레임을 처음에는 숨김
        self.manual_params_frame.grid_remove()
        
        # 탐지 방법 선택 (통합 시스템)
        ttk.Label(param_frame, text="탐지 방법:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.method_var = tk.StringVar(value="auto")
        method_combo = ttk.Combobox(param_frame, textvariable=self.method_var, 
                                   values=["auto", "simple", "window", "slope", "alternating", "enhanced", "strict"], 
                                   state="readonly", width=15)
        method_combo.grid(row=2, column=1, padx=(10, 0), pady=(5, 0))
        
        # 방법 설명 라벨
        method_desc = ttk.Label(param_frame, text="(auto: 자동 선택, 다른 방법: 수동 선택)", 
                               font=('Arial', 8), foreground='gray')
        method_desc.grid(row=2, column=2, sticky=tk.W, padx=(5, 0), pady=(5, 0))
        
        # 윈도우 크기 설정 (improved_window 방법일 때만)
        ttk.Label(param_frame, text="윈도우 크기:").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        self.window_size_var = tk.StringVar(value="3")
        self.window_size_entry = ttk.Entry(param_frame, textvariable=self.window_size_var, width=10)
        self.window_size_entry.grid(row=3, column=1, padx=(10, 0), pady=(5, 0))
        
        # 임계값 설정
        ttk.Label(param_frame, text="임계값:").grid(row=4, column=0, sticky=tk.W, pady=(5, 0))
        self.threshold_var = tk.StringVar(value="0.0001")
        threshold_entry = ttk.Entry(param_frame, textvariable=self.threshold_var, width=10)
        threshold_entry.grid(row=4, column=1, padx=(10, 0), pady=(5, 0))
        
        # 최대값 임계값 설정
        ttk.Label(param_frame, text="최대값 임계값:").grid(row=5, column=0, sticky=tk.W, pady=(5, 0))
        self.max_threshold_var = tk.StringVar(value="0.0")
        max_threshold_entry = ttk.Entry(param_frame, textvariable=self.max_threshold_var, width=10)
        max_threshold_entry.grid(row=5, column=1, padx=(10, 0), pady=(5, 0))
        ttk.Label(param_frame, text="이상 (0이면 제한 없음)").grid(row=5, column=2, sticky=tk.W, padx=(5, 0), pady=(5, 0))
        
        # 최소값 임계값 설정
        ttk.Label(param_frame, text="최소값 임계값:").grid(row=6, column=0, sticky=tk.W, pady=(5, 0))
        self.min_threshold_var = tk.StringVar(value="1.0")
        min_threshold_entry = ttk.Entry(param_frame, textvariable=self.min_threshold_var, width=10)
        min_threshold_entry.grid(row=6, column=1, padx=(10, 0), pady=(5, 0))
        ttk.Label(param_frame, text="이하 (999이면 제한 없음)").grid(row=6, column=2, sticky=tk.W, padx=(5, 0), pady=(5, 0))
        
        # 분석 실행 버튼
        analyze_button = ttk.Button(left_frame, text="3. 분석 실행", 
                                   command=self.run_analysis, style='Accent.TButton')
        analyze_button.grid(row=3, column=0, columnspan=3, pady=(0, 10))
        
        # 결과 표시 섹션
        result_frame = ttk.LabelFrame(left_frame, text="4. 분석 결과", padding="10")
        result_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # 결과 텍스트 영역
        self.result_text = scrolledtext.ScrolledText(result_frame, height=15, width=70)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 스크롤바
        result_scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_text.yview)
        result_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.result_text.configure(yscrollcommand=result_scrollbar.set)
        
        # 저장 버튼들
        button_frame = ttk.Frame(result_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(button_frame, text="결과 저장", 
                  command=self.save_results).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="그래프 저장", 
                  command=self.save_graph).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="차이값 계산", 
                  command=self.calculate_differences).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="차이값 저장", 
                  command=self.save_differences).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="결과 지우기", 
                  command=self.clear_results).pack(side=tk.LEFT)
        
        # 그래프 영역 설정
        self.setup_graph_area(right_frame)
        
        # 그리드 가중치 설정
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)  # 그래프 영역이 더 크게
        main_frame.rowconfigure(0, weight=1)
        left_frame.rowconfigure(4, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        
        # 초기 메시지 표시
        self.show_initial_message()
    
    def save_graph(self):
        """현재 그래프를 이미지 파일로 저장합니다."""
        if not self.figure:
            messagebox.showwarning("경고", "저장할 그래프가 없습니다.")
            return
        
        # 파일 저장 다이얼로그
        file_path = filedialog.asksaveasfilename(
            title="그래프 저장",
            defaultextension=".png",
            filetypes=[
                ("PNG 이미지", "*.png"),
                ("JPG 이미지", "*.jpg"),
                ("PDF 문서", "*.pdf"),
                ("SVG 벡터", "*.svg"),
                ("모든 파일", "*.*")
            ]
        )
        
        if file_path:
            try:
                # 그래프 저장 (DPI 높게 설정하여 고해상도로)
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight', 
                                  facecolor='white', edgecolor='none')
                
                messagebox.showinfo("성공", f"그래프가 저장되었습니다:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("오류", f"그래프 저장 중 오류가 발생했습니다:\n{str(e)}")
    
    def setup_graph_area(self, parent_frame):
        """그래프 영역을 설정합니다."""
        # matplotlib figure 생성
        self.figure = plt.Figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        
        # 캔버스 생성 및 배치
        self.canvas = FigureCanvasTkAgg(self.figure, parent_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 초기 그래프 설정
        self.ax.set_title('Data Graph')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.grid(True, alpha=0.3)
        
        # 마우스 호버 이벤트 설정
        self.setup_hover_events()
    
    def setup_hover_events(self):
        """마우스 호버 이벤트를 설정합니다."""
        # 호버 상태를 저장할 변수들
        self.hover_annotation = None
        self.current_hover_point = None
        
        # 마우스 이벤트 연결
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.canvas.mpl_connect("axes_leave_event", self.on_leave)
        self.canvas.mpl_connect("button_press_event", self.on_click)
    
    def on_hover(self, event):
        """마우스 호버 이벤트 핸들러"""
        if event.inaxes != self.ax:
            return
        
        # 기존 호버 주석 제거
        if self.hover_annotation:
            self.hover_annotation.remove()
            self.hover_annotation = None
        
        # 데이터가 없으면 리턴
        if not hasattr(self, 'x_data') or not self.x_data or not self.y_data:
            return
        
        # 마우스 위치에서 가장 가까운 데이터 포인트 찾기
        if event.xdata is not None and event.ydata is not None:
            # 모든 데이터 포인트와의 거리 계산
            distances = []
            for i, (x, y) in enumerate(zip(self.x_data, self.y_data)):
                dist = np.sqrt((event.xdata - x)**2 + (event.ydata - y)**2)
                distances.append((dist, i, x, y))
            
            # 가장 가까운 포인트 찾기
            if distances:
                min_dist, idx, x_val, y_val = min(distances)
                
                # 호버 반경 설정 (화면 좌표 기준)
                if min_dist < 0.05:  # 적절한 호버 반경
                    # 호버 주석 생성
                    self.hover_annotation = self.ax.annotate(
                        f'X: {x_val:.4f}\nY: {y_val:.4f}',
                        xy=(x_val, y_val),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                        zorder=10
                    )
                    self.current_hover_point = (x_val, y_val)
                    
                    # 캔버스 업데이트
                    self.canvas.draw_idle()
    
    def on_leave(self, event):
        """마우스가 그래프 영역을 벗어날 때 호출"""
        if self.hover_annotation:
            self.hover_annotation.remove()
            self.hover_annotation = None
            self.current_hover_point = None
            self.canvas.draw_idle()
    
    def toggle_mode(self):
        """탐지 모드를 전환합니다."""
        mode = self.mode_var.get()
        if mode == "auto":
            self.auto_params_frame.grid()
            self.manual_params_frame.grid_remove()
            self.interactive_mode = False
        else:  # manual
            self.auto_params_frame.grid_remove()
            self.manual_params_frame.grid()
            self.interactive_mode = True
            self.update_selection_info()
    
    def on_click(self, event):
        """마우스 클릭 이벤트 핸들러"""
        if not self.interactive_mode or event.inaxes != self.ax:
            return
        
        if not hasattr(self, 'x_data') or not self.x_data or not self.y_data:
            return
        
        if event.xdata is not None and event.ydata is not None:
            # 가장 가까운 데이터 포인트 찾기
            distances = []
            for i, (x, y) in enumerate(zip(self.x_data, self.y_data)):
                dist = np.sqrt((event.xdata - x)**2 + (event.ydata - y)**2)
                distances.append((dist, i, x, y))
            
            if distances:
                min_dist, idx, x_val, y_val = min(distances)
                
                # 클릭 반경 설정
                if min_dist < 0.05:
                    if event.button == 1:  # 좌클릭 - 최대값 선택
                        self.add_maximum(idx, x_val, y_val)
                    elif event.button == 3:  # 우클릭 - 최소값 선택
                        self.add_minimum(idx, x_val, y_val)
    
    def add_maximum(self, idx, x_val, y_val):
        """최대값을 추가합니다."""
        # 중복 확인
        for existing_idx, _, _ in self.selected_maxima:
            if existing_idx == idx:
                return
        
        self.selected_maxima.append((idx, x_val, y_val))
        self.update_selection_info()
        self.update_graph_with_selections()
    
    def add_minimum(self, idx, x_val, y_val):
        """최소값을 추가합니다."""
        # 중복 확인
        for existing_idx, _, _ in self.selected_minima:
            if existing_idx == idx:
                return
        
        self.selected_minima.append((idx, x_val, y_val))
        self.update_selection_info()
        self.update_graph_with_selections()
    
    def clear_selections(self):
        """선택된 점들을 초기화합니다."""
        self.selected_maxima = []
        self.selected_minima = []
        self.update_selection_info()
        self.update_graph_with_selections()
    
    def update_selection_info(self):
        """선택 정보를 업데이트합니다."""
        if hasattr(self, 'selection_info'):
            self.selection_info.config(
                text=f"선택된 최대값: {len(self.selected_maxima)}개, 최소값: {len(self.selected_minima)}개"
            )
    
    def update_graph_with_selections(self):
        """선택된 점들을 그래프에 표시합니다."""
        if not hasattr(self, 'ax') or not self.ax:
            return
        
        # 기존 선택 표시 제거
        for artist in self.ax.collections[:]:
            if hasattr(artist, '_is_selection'):
                artist.remove()
        
        # 선택된 최대값 표시
        if self.selected_maxima:
            max_x = [x for _, x, _ in self.selected_maxima]
            max_y = [y for _, _, y in self.selected_maxima]
            scatter = self.ax.scatter(max_x, max_y, s=100, c='red', marker='^', 
                                    alpha=0.9, edgecolors='darkred', linewidth=2)
            scatter._is_selection = True
        
        # 선택된 최소값 표시
        if self.selected_minima:
            min_x = [x for _, x, _ in self.selected_minima]
            min_y = [y for _, _, y in self.selected_minima]
            scatter = self.ax.scatter(min_x, min_y, s=100, c='green', marker='v', 
                                    alpha=0.9, edgecolors='darkgreen', linewidth=2)
            scatter._is_selection = True
        
        self.canvas.draw_idle()
    
    def show_initial_message(self):
        """초기 안내 메시지를 표시합니다."""
        message = """통합된 만능 로컬 극값 탐지 프로그램에 오신 것을 환영합니다!

🎯 새로운 기능:
- 데이터 특성을 자동으로 분석하여 최적의 탐지 방법을 선택합니다
- 6가지 탐지 방법을 하나로 통합했습니다
- 더 정확하고 신뢰할 수 있는 극값 탐지가 가능합니다
- 마우스 호버로 그래프의 모든 점의 좌표를 확인할 수 있습니다
- 인터랙티브 수동 선택 모드: 그래프에서 직접 클릭하여 극값을 선택할 수 있습니다
- 차이값 계산: 최대값과 최소값을 번호 순으로 매칭하여 차이값을 계산하고 저장할 수 있습니다

사용 방법:

📊 자동 탐지 모드:
1. '파일 선택' 버튼을 클릭하여 데이터 파일(.txt)을 선택하세요
2. '자동 탐지' 모드를 선택하세요
3. 원하는 파라미터를 설정하세요
4. '분석 실행' 버튼을 클릭하여 분석을 시작하세요

🎯 수동 선택 모드:
1. '파일 선택' 버튼을 클릭하여 데이터 파일(.txt)을 선택하세요
2. '수동 선택' 모드를 선택하세요
3. 그래프에서 좌클릭으로 최대값 후보를, 우클릭으로 최소값 후보를 선택하세요
4. '분석 실행' 버튼을 클릭하여 선택된 점 주변의 실제 극값을 찾으세요

💡 공통 기능:
- 그래프에서 마우스를 움직여 각 점의 좌표를 확인하세요
- 결과를 확인하고 필요시 저장하세요
- '차이값 계산' 버튼으로 최대값-최소값 차이를 계산하세요
- '차이값 저장' 버튼으로 차이값 결과를 파일로 저장하세요

지원하는 파일 형식: 텍스트 파일 (.txt)
각 줄에 하나의 숫자(실수)가 있어야 합니다.

탐지 방법 설명:
- auto: 데이터 특성에 따라 자동으로 최적의 방법 선택 (권장)
- simple: 단순한 이웃 비교 방법
- window: 윈도우 기반 방법
- slope: 기울기 변화 기반 방법
- alternating: 교차 패턴 보장 방법
- enhanced: 향상된 교차 패턴 방법
- strict: 엄격한 조건 적용 방법
"""
        self.result_text.insert(tk.END, message)
    
    def select_file(self):
        """파일 선택 다이얼로그를 열고 파일을 로드합니다."""
        file_path = filedialog.askopenfilename(
            title="데이터 파일 선택",
            filetypes=[("텍스트 파일", "*.txt"), ("모든 파일", "*.*")]
        )
        
        if file_path:
            self.load_file(file_path)
    
    def parse_xy_data(self, file_path):
        """X-Y 좌표 데이터를 파싱합니다."""
        x_data = []
        y_data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            # X, Y 라벨 찾기
            x_start = -1
            y_start = -1
            
            for i, line in enumerate(lines):
                line = line.strip()
                if line == 'X' or line == 'x':
                    x_start = i + 1
                elif line == 'Y' or line == 'y':
                    y_start = i + 1
            
            # X 데이터 파싱
            if x_start >= 0:
                for i in range(x_start, len(lines)):
                    line = lines[i].strip()
                    if line == 'Y' or line == 'y':
                        break
                    if line:
                        try:
                            x_data.append(float(line))
                        except ValueError:
                            pass
            
            # Y 데이터 파싱
            if y_start >= 0:
                for i in range(y_start, len(lines)):
                    line = lines[i].strip()
                    if line:
                        try:
                            y_data.append(float(line))
                        except ValueError:
                            pass
            
            # X 라벨이 없으면 인덱스를 X로 사용
            if not x_data and y_data:
                x_data = list(range(len(y_data)))
            
            # 데이터 길이 맞추기
            min_len = min(len(x_data), len(y_data))
            if min_len > 0:
                x_data = x_data[:min_len]
                y_data = y_data[:min_len]
            
            return x_data, y_data
            
        except Exception as e:
            raise Exception(f"X-Y 데이터 파싱 중 오류: {str(e)}")
    
    def load_file(self, file_path):
        """선택된 파일을 로드하고 데이터를 읽습니다."""
        try:
            # X-Y 데이터 파싱 시도
            self.x_data, self.y_data = self.parse_xy_data(file_path)
            
            if not self.y_data:
                messagebox.showerror("오류", "파일에서 유효한 데이터를 찾을 수 없습니다.")
                return
            
            # Y 데이터를 data로 설정 (기존 코드 호환성)
            self.data = self.y_data.copy()
            
            self.current_file_path = file_path
            filename = os.path.basename(file_path)
            self.file_label.config(text=f"선택된 파일: {filename}")
            
            # 파일 정보 표시
            file_info = f"X 데이터: {len(self.x_data)}개 | Y 데이터: {len(self.y_data)}개 | 최대값: {max(self.y_data):.6f} | 최소값: {min(self.y_data):.6f}"
            self.file_info_label.config(text=file_info)
            
            # 그래프 그리기
            self.plot_data()
            
            messagebox.showinfo("성공", f"파일이 성공적으로 로드되었습니다.\nX 데이터: {len(self.x_data)}개, Y 데이터: {len(self.y_data)}개")
            
        except FileNotFoundError:
            messagebox.showerror("오류", "파일을 찾을 수 없습니다.")
        except Exception as e:
            messagebox.showerror("오류", f"파일을 읽는 중 오류가 발생했습니다:\n{str(e)}")
    
    def plot_data(self):
        """데이터를 그래프로 그립니다."""
        if not self.x_data or not self.y_data:
            return
        
        # 그래프 클리어
        self.ax.clear()
        
        # 데이터 플롯
        self.ax.plot(self.x_data, self.y_data, 'b-', linewidth=1, alpha=0.7, label='Data')
        self.ax.scatter(self.x_data, self.y_data, s=1, c='blue', alpha=0.5)
        
        # 그래프 설정
        self.ax.set_title(f'Data Graph - {os.path.basename(self.current_file_path)}')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # 캔버스 업데이트
        self.canvas.draw()
    
    def plot_extrema(self, minima, maxima):
        """극값을 그래프에 표시합니다."""
        if not self.x_data or not self.y_data:
            return
        
        # 그래프 클리어 후 다시 그리기
        self.ax.clear()
        
        # 원본 데이터 플롯
        self.ax.plot(self.x_data, self.y_data, 'b-', linewidth=1, alpha=0.7, label='Data')
        self.ax.scatter(self.x_data, self.y_data, s=1, c='blue', alpha=0.5)
        
        # 극값 표시 (호버로 좌표 확인 가능)
        if maxima:
            max_x = [self.x_data[idx] for idx, _ in maxima]
            max_y = [val for _, val in maxima]
            self.ax.scatter(max_x, max_y, s=50, c='red', marker='^', label=f'Maxima ({len(maxima)})', alpha=0.8)
        
        if minima:
            min_x = [self.x_data[idx] for idx, _ in minima]
            min_y = [val for _, val in minima]
            self.ax.scatter(min_x, min_y, s=50, c='green', marker='v', label=f'Minima ({len(minima)})', alpha=0.8)
        
        # 그래프 설정
        self.ax.set_title(f'Local Extrema Analysis - {os.path.basename(self.current_file_path)}')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # 캔버스 업데이트
        self.canvas.draw()
    
    def get_user_constraints(self):
        """사용자가 설정한 제약 조건을 가져옵니다."""
        try:
            max_count = int(self.max_count_var.get())
            min_count = int(self.min_count_var.get())
            
            if max_count < 0 or min_count < 0:
                raise ValueError("개수는 0 이상이어야 합니다.")
            
            return max_count, min_count
        except ValueError as e:
            messagebox.showerror("오류", f"잘못된 입력값입니다: {str(e)}")
            return None, None
    
    def find_local_extrema_unified(self, data, method="auto", **kwargs):
        """통합된 탐지 시스템을 사용하여 로컬 극값을 찾습니다."""
        if method == "auto":
            return self.detector.detect_extrema(data, **kwargs)
        else:
            return self.detector.detect_extrema(data, method=method, **kwargs)
    
    
    def apply_threshold_filter(self, minima, maxima, max_threshold, min_threshold):
        """
        최대값과 최소값에 임계값 필터링을 적용합니다.
        
        Args:
            minima: 최소값 리스트 [(인덱스, 값), ...]
            maxima: 최대값 리스트 [(인덱스, 값), ...]
            max_threshold: 최대값 임계값 (이상이어야 함)
            min_threshold: 최소값 임계값 (이하여야 함)
        
        Returns:
            필터링된 (minima, maxima) 튜플
        """
        # 최대값 필터링: max_threshold 이상인 값만 유지
        if max_threshold > 0:
            filtered_maxima = [(idx, val) for idx, val in maxima if val >= max_threshold]
        else:
            filtered_maxima = maxima
        
        # 최소값 필터링: min_threshold 이하인 값만 유지 (999는 제한 없음을 의미)
        if min_threshold < 999:
            filtered_minima = [(idx, val) for idx, val in minima if val <= min_threshold]
        else:
            filtered_minima = minima
        
        return filtered_minima, filtered_maxima
    
    def filter_results_by_count(self, minima, maxima, max_count_limit, min_count_limit, method="auto"):
        """결과를 개수 제한에 따라 필터링합니다."""
        # alternating 계열 방법에서는 인덱스 순서를 유지하여 교차 패턴 보장
        if method in ["alternating", "enhanced", "strict"]:
            # alternating 계열 방법에서는 개수 제한을 무시하고 모든 극값 반환
            return minima, maxima
        
        # 다른 방법들에서는 기존 방식 사용
        # 최대값을 절댓값 기준으로 정렬하여 상위 N개 선택
        if max_count_limit > 0 and len(maxima) > max_count_limit:
            maxima_sorted = sorted(maxima, key=lambda x: abs(x[1]), reverse=True)
            maxima = maxima_sorted[:max_count_limit]
        
        # 최소값을 절댓값 기준으로 정렬하여 상위 N개 선택
        if min_count_limit > 0 and len(minima) > min_count_limit:
            minima_sorted = sorted(minima, key=lambda x: abs(x[1]), reverse=True)
            minima = minima_sorted[:min_count_limit]
        
        return minima, maxima
    
    def run_analysis(self):
        """분석을 실행합니다."""
        if not self.data:
            messagebox.showerror("오류", "먼저 데이터 파일을 선택해주세요.")
            return
        
        # 모드 확인
        mode = self.mode_var.get()
        
        # 분석 실행
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "분석을 시작합니다...\n\n")
        self.root.update()
        
        try:
            if mode == "manual":
                # 수동 선택 모드
                if not self.selected_maxima and not self.selected_minima:
                    messagebox.showwarning("경고", "수동 선택 모드에서는 최소 하나의 점을 선택해주세요.")
                    return
                
                # 선택된 점들을 기반으로 인근 극값 탐지
                minima, maxima = self.find_extrema_around_selections()
                
                # 결과 저장
                self.results = {
                    'minima': minima,
                    'maxima': maxima,
                    'method': 'manual_selection',
                    'file': os.path.basename(self.current_file_path)
                }
                
                # 결과 표시
                self.display_manual_results(minima, maxima)
                
                # 그래프에 극값 표시
                self.plot_extrema(minima, maxima)
                
            else:
                # 자동 탐지 모드
                # 사용자 제약 조건 가져오기
                max_count_limit, min_count_limit = self.get_user_constraints()
                if max_count_limit is None:
                    return
                
                # 파라미터 가져오기
                method = self.method_var.get()
                
                try:
                    threshold = float(self.threshold_var.get())
                    window_size = int(self.window_size_var.get())
                    max_threshold = float(self.max_threshold_var.get())
                    min_threshold = float(self.min_threshold_var.get())
                except ValueError:
                    messagebox.showerror("오류", "모든 임계값과 윈도우 크기는 숫자여야 합니다.")
                    return
                
                # 통합된 탐지 시스템 사용
                if method == "auto":
                    minima, maxima = self.find_local_extrema_unified(self.data, method="auto", 
                                                                   threshold=threshold, window_size=window_size)
                else:
                    minima, maxima = self.find_local_extrema_unified(self.data, method=method, 
                                                                   threshold=threshold, window_size=window_size)
                
                # 모든 방법에 대해 임계값 필터링 적용
                minima, maxima = self.apply_threshold_filter(minima, maxima, max_threshold, min_threshold)
                
                # 결과 필터링
                minima, maxima = self.filter_results_by_count(minima, maxima, max_count_limit, min_count_limit, method)
                
                # 결과 저장
                self.results = {
                    'minima': minima,
                    'maxima': maxima,
                    'method': method,
                    'file': os.path.basename(self.current_file_path)
                }
                
                # 결과 표시
                self.display_results(minima, maxima, method, max_count_limit, min_count_limit, max_threshold, min_threshold)
                
                # 그래프에 극값 표시
                self.plot_extrema(minima, maxima)
            
        except Exception as e:
            messagebox.showerror("오류", f"분석 중 오류가 발생했습니다:\n{str(e)}")
            self.result_text.insert(tk.END, f"오류 발생: {str(e)}\n")
    
    def display_results(self, minima, maxima, method, max_count_limit, min_count_limit, max_threshold, min_threshold):
        """분석 결과를 표시합니다."""
        # 탐지기 정보 가져오기
        detector_info = self.detector.get_detection_info()
        data_stats = detector_info.get('data_stats', {})
        
        result_text = f"""
=== 분석 결과 ===
파일: {os.path.basename(self.current_file_path)}
데이터 개수: {len(self.data)}개
탐지 방법: {method} {'(자동 선택)' if method == 'auto' else '(수동 선택)'}
최대값 개수 제한: {max_count_limit if max_count_limit > 0 else '제한 없음'}개
최소값 개수 제한: {min_count_limit if min_count_limit > 0 else '제한 없음'}개
최대값 임계값: {max_threshold:.3f} 이상 (0이면 제한 없음)
최소값 임계값: {min_threshold:.3f} 이하 (999이면 제한 없음)

=== 데이터 특성 분석 ===
데이터 변동성: {data_stats.get('variability', 0):.4f}
노이즈 레벨: {data_stats.get('noise_level', 0):.6f}
극값 밀도: {data_stats.get('extrema_density', 0):.4f}
진동 패턴: {'예' if data_stats.get('is_oscillatory', False) else '아니오'}
평평한 구간: {'예' if data_stats.get('has_plateaus', False) else '아니오'}

=== 발견된 극값 ===
로컬 최대값: {len(maxima)}개
로컬 최소값: {len(minima)}개

=== 로컬 최대값들 ===
"""
        
        if maxima:
            for i, (idx, val) in enumerate(maxima, 1):
                result_text += f"{i:3d}. 인덱스: {idx:4d}, 값: {val:12.8f}\n"
        else:
            result_text += "발견된 최대값이 없습니다.\n"
        
        result_text += "\n=== 로컬 최소값들 ===\n"
        
        if minima:
            for i, (idx, val) in enumerate(minima, 1):
                result_text += f"{i:3d}. 인덱스: {idx:4d}, 값: {val:12.8f}\n"
        else:
            result_text += "발견된 최소값이 없습니다.\n"
        
        # 데이터 통계
        result_text += f"""
=== 데이터 통계 ===
전체 최대값: {max(self.data):.8f}
전체 최소값: {min(self.data):.8f}
전체 평균값: {sum(self.data)/len(self.data):.8f}
"""
        
        self.result_text.insert(tk.END, result_text)
    
    def find_extrema_around_selections(self):
        """선택된 점들을 기반으로 인근 극값을 탐지합니다."""
        minima = []
        maxima = []
        
        # 선택된 최대값 주변에서 실제 최대값 찾기
        for selected_idx, selected_x, selected_y in self.selected_maxima:
            # 주변 윈도우에서 최대값 찾기
            window_size = 10  # 주변 10개 점 확인
            start_idx = max(0, selected_idx - window_size)
            end_idx = min(len(self.data), selected_idx + window_size + 1)
            
            local_max_idx = selected_idx
            local_max_val = self.data[selected_idx]
            
            # 주변에서 실제 최대값 찾기
            for i in range(start_idx, end_idx):
                if self.data[i] > local_max_val:
                    local_max_idx = i
                    local_max_val = self.data[i]
            
            # 최대값으로 추가
            if (local_max_idx, local_max_val) not in maxima:
                maxima.append((local_max_idx, local_max_val))
        
        # 선택된 최소값 주변에서 실제 최소값 찾기
        for selected_idx, selected_x, selected_y in self.selected_minima:
            # 주변 윈도우에서 최소값 찾기
            window_size = 10  # 주변 10개 점 확인
            start_idx = max(0, selected_idx - window_size)
            end_idx = min(len(self.data), selected_idx + window_size + 1)
            
            local_min_idx = selected_idx
            local_min_val = self.data[selected_idx]
            
            # 주변에서 실제 최소값 찾기
            for i in range(start_idx, end_idx):
                if self.data[i] < local_min_val:
                    local_min_idx = i
                    local_min_val = self.data[i]
            
            # 최소값으로 추가
            if (local_min_idx, local_min_val) not in minima:
                minima.append((local_min_idx, local_min_val))
        
        return minima, maxima
    
    def display_manual_results(self, minima, maxima):
        """수동 선택 모드의 분석 결과를 표시합니다."""
        result_text = f"""
=== 수동 선택 모드 분석 결과 ===
파일: {os.path.basename(self.current_file_path)}
데이터 개수: {len(self.data)}개
탐지 방법: 수동 선택 + 인근 극값 탐지

=== 사용자 선택 ===
선택된 최대값 후보: {len(self.selected_maxima)}개
선택된 최소값 후보: {len(self.selected_minima)}개

=== 발견된 극값 ===
로컬 최대값: {len(maxima)}개
로컬 최소값: {len(minima)}개

=== 로컬 최대값들 ===
"""
        
        if maxima:
            for i, (idx, val) in enumerate(maxima, 1):
                result_text += f"{i:3d}. 인덱스: {idx:4d}, 값: {val:12.8f}\n"
        else:
            result_text += "발견된 최대값이 없습니다.\n"
        
        result_text += "\n=== 로컬 최소값들 ===\n"
        
        if minima:
            for i, (idx, val) in enumerate(minima, 1):
                result_text += f"{i:3d}. 인덱스: {idx:4d}, 값: {val:12.8f}\n"
        else:
            result_text += "발견된 최소값이 없습니다.\n"
        
        # 데이터 통계
        result_text += f"""
=== 데이터 통계 ===
전체 최대값: {max(self.data):.8f}
전체 최소값: {min(self.data):.8f}
전체 평균값: {sum(self.data)/len(self.data):.8f}
"""
        
        self.result_text.insert(tk.END, result_text)
    
    def calculate_differences(self):
        """최대값과 최소값의 차이를 계산합니다."""
        if not self.results or not self.results.get('maxima') or not self.results.get('minima'):
            messagebox.showwarning("경고", "먼저 분석을 실행하여 최대값과 최소값을 구해주세요.")
            return
        
        maxima = self.results['maxima']
        minima = self.results['minima']
        
        # 최대값과 최소값을 인덱스 순으로 정렬
        maxima_sorted = sorted(maxima, key=lambda x: x[0])  # 인덱스 기준 정렬
        minima_sorted = sorted(minima, key=lambda x: x[0])  # 인덱스 기준 정렬
        
        # 차이값 계산
        self.difference_results = []
        min_len = min(len(maxima_sorted), len(minima_sorted))
        
        for i in range(min_len):
            max_idx, max_val = maxima_sorted[i]
            min_idx, min_val = minima_sorted[i]
            difference = max_val - min_val
            
            self.difference_results.append({
                'pair_number': i + 1,
                'max_index': max_idx,
                'max_value': max_val,
                'min_index': min_idx,
                'min_value': min_val,
                'difference': difference
            })
        
        # 결과 표시
        self.display_difference_results()
        
        messagebox.showinfo("성공", f"{len(self.difference_results)}개의 차이값이 계산되었습니다.")
    
    def display_difference_results(self):
        """차이값 계산 결과를 표시합니다."""
        if not self.difference_results:
            return
        
        result_text = f"""
=== 차이값 계산 결과 ===
총 {len(self.difference_results)}개의 최대값-최소값 쌍이 계산되었습니다.

"""
        
        for result in self.difference_results:
            result_text += f"쌍 {result['pair_number']:2d}: 최대값({result['max_index']:4d}, {result['max_value']:10.6f}) - 최소값({result['min_index']:4d}, {result['min_value']:10.6f}) = {result['difference']:10.6f}\n"
        
        # 통계 정보
        differences = [r['difference'] for r in self.difference_results]
        result_text += f"""
=== 차이값 통계 ===
평균 차이값: {sum(differences)/len(differences):.6f}
최대 차이값: {max(differences):.6f}
최소 차이값: {min(differences):.6f}
차이값 표준편차: {(sum([(d - sum(differences)/len(differences))**2 for d in differences]) / len(differences))**0.5:.6f}

"""
        
        self.result_text.insert(tk.END, result_text)
    
    def save_differences(self):
        """차이값 계산 결과를 파일로 저장합니다."""
        if not self.difference_results:
            messagebox.showwarning("경고", "저장할 차이값 결과가 없습니다. 먼저 '차이값 계산' 버튼을 클릭해주세요.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="차이값 결과 저장",
            defaultextension=".txt",
            filetypes=[("텍스트 파일", "*.txt"), ("CSV 파일", "*.csv"), ("모든 파일", "*.*")]
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    # CSV 형식으로 저장
                    import csv
                    with open(file_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['쌍 번호', '최대값 인덱스', '최대값', '최소값 인덱스', '최소값', '차이값'])
                        for result in self.difference_results:
                            writer.writerow([
                                result['pair_number'],
                                result['max_index'],
                                result['max_value'],
                                result['min_index'],
                                result['min_value'],
                                result['difference']
                            ])
                else:
                    # 텍스트 형식으로 저장
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write("=== 최대값-최소값 차이값 계산 결과 ===\n")
                        f.write(f"파일: {os.path.basename(self.current_file_path)}\n")
                        f.write(f"총 쌍 개수: {len(self.difference_results)}개\n\n")
                        
                        for result in self.difference_results:
                            f.write(f"쌍 {result['pair_number']:2d}: 최대값({result['max_index']:4d}, {result['max_value']:10.6f}) - 최소값({result['min_index']:4d}, {result['min_value']:10.6f}) = {result['difference']:10.6f}\n")
                        
                        # 통계 정보
                        differences = [r['difference'] for r in self.difference_results]
                        f.write(f"\n=== 차이값 통계 ===\n")
                        f.write(f"평균 차이값: {sum(differences)/len(differences):.6f}\n")
                        f.write(f"최대 차이값: {max(differences):.6f}\n")
                        f.write(f"최소 차이값: {min(differences):.6f}\n")
                        f.write(f"차이값 표준편차: {(sum([(d - sum(differences)/len(differences))**2 for d in differences]) / len(differences))**0.5:.6f}\n")
                
                messagebox.showinfo("성공", f"차이값 결과가 저장되었습니다:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("오류", f"파일 저장 중 오류가 발생했습니다:\n{str(e)}")
    
    def save_results(self):
        """분석 결과를 파일로 저장합니다."""
        if not self.results:
            messagebox.showwarning("경고", "저장할 결과가 없습니다. 먼저 분석을 실행해주세요.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="결과 저장",
            defaultextension=".txt",
            filetypes=[("텍스트 파일", "*.txt"), ("모든 파일", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"=== 로컬 극값 탐지 결과 ===\n")
                    f.write(f"파일: {self.results['file']}\n")
                    f.write(f"탐지 방법: {self.results['method']}\n")
                    f.write(f"로컬 최대값 개수: {len(self.results['maxima'])}\n")
                    f.write(f"로컬 최소값 개수: {len(self.results['minima'])}\n\n")
                    
                    if self.results['maxima']:
                        f.write("로컬 최대값들:\n")
                        for i, (idx, val) in enumerate(self.results['maxima'], 1):
                            f.write(f"{i:3d}. 인덱스: {idx:4d}, 값: {val:12.8f}\n")
                    
                    if self.results['minima']:
                        f.write("\n로컬 최소값들:\n")
                        for i, (idx, val) in enumerate(self.results['minima'], 1):
                            f.write(f"{i:3d}. 인덱스: {idx:4d}, 값: {val:12.8f}\n")
                
                messagebox.showinfo("성공", f"결과가 저장되었습니다:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("오류", f"파일 저장 중 오류가 발생했습니다:\n{str(e)}")
    
    def clear_results(self):
        """결과를 지웁니다."""
        self.result_text.delete(1.0, tk.END)
        self.results = {}
        self.difference_results = []  # 차이값 결과도 초기화
        
        # 그래프도 원본 데이터로 초기화
        if self.x_data and self.y_data:
            self.plot_data()
        
        self.show_initial_message()


def main():
    """메인 함수"""
    root = tk.Tk()
    app = LocalExtremaGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
