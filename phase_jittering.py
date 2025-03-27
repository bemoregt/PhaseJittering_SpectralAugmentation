import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class PhaseJitteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phase Jittering Image Augmentation")
        
        # 메인 프레임 생성
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)
        
        # 버튼 생성
        self.load_button = tk.Button(self.main_frame, text="이미지 로드", command=self.load_image)
        self.load_button.pack(pady=5)
        
        self.process_button = tk.Button(self.main_frame, text="위상 지터링 적용", command=self.process_image)
        self.process_button.pack(pady=5)
        self.process_button.config(state='disabled')
        
        # 이미지 표시 영역
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.original_image = None
        self.f_shift = None
        self.augmented_images = []  # 지터링된 이미지들을 저장할 리스트
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        
        if file_path:
            self.original_image = Image.open(file_path).convert("L")
            img_array = np.array(self.original_image)
            
            f_transform = np.fft.fft2(img_array)
            self.f_shift = np.fft.fftshift(f_transform)
            
            self.display_original_image()
            self.process_button.config(state='normal')
    
    def apply_phase_jittering(self, f_shift, jitter_amount):
        phase_random = np.exp(1j * 2 * np.pi * np.random.rand(*f_shift.shape) * jitter_amount)
        f_jittered = f_shift * phase_random
        f_ishift = np.fft.ifftshift(f_jittered)
        img_jittered = np.fft.ifft2(f_ishift)
        img_jittered = np.abs(img_jittered)
        
        # 이미지 정규화 추가
        img_jittered = (img_jittered - np.min(img_jittered)) / (np.max(img_jittered) - np.min(img_jittered))
        return img_jittered

    def display_original_image(self):
        # 원본 이미지 표시 시에도 정규화된 배열로 변환
        img_array = np.array(self.original_image)
        img_normalized = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
        resized_image = Image.fromarray((img_normalized * 255).astype(np.uint8)).resize((400, 400))
        photo = ImageTk.PhotoImage(resized_image)
        if hasattr(self, 'image_label'):
            self.image_label.destroy()
        self.image_label = tk.Label(self.canvas_frame, image=photo)
        self.image_label.image = photo
        self.image_label.pack(pady=10)
    
    def on_click(self, event):
        if event.inaxes is not None:
            # 클릭된 축의 인덱스 찾기
            clicked_index = list(self.fig.axes).index(event.inaxes)
            if clicked_index < len(self.augmented_images):
                # 클릭된 이미지를 PIL Image로 변환
                clicked_image = Image.fromarray(
                    (self.augmented_images[clicked_index] * 255).astype(np.uint8)
                )
                # 이미지 크기 조정 및 표시
                resized_image = clicked_image.resize((400, 400))
                photo = ImageTk.PhotoImage(resized_image)
                self.image_label.configure(image=photo)
                self.image_label.image = photo
    
    def process_image(self):
        if self.f_shift is None:
            return
            
        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()
            
        self.fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        self.fig.suptitle("Phase Jittering Results (Click to Enlarge)")
        
        self.augmented_images = []  # 리스트 초기화
        ## ------------------------------------------------------------------
        jitter_amounts = np.linspace(0.01, 0.9, 5)
        
        for ax, jitter in zip(axes, jitter_amounts):
            augmented = self.apply_phase_jittering(self.f_shift, jitter)
            self.augmented_images.append(augmented)  # 이미지 저장
            ax.imshow(augmented, cmap='gray')
            ax.set_title(f'Jitter: {jitter:.2f}')
            ax.axis('off')
            
        plt.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(pady=10)
        
        # 클릭 이벤트 연결
        self.canvas.mpl_connect('button_press_event', self.on_click)

if __name__ == "__main__":
    root = tk.Tk()
    app = PhaseJitteringApp(root)
    root.mainloop()