"""
Микросервис для анализа данных толщинометрии труб.
Реализует функции контроллера ThickController из Java Spring Boot приложения.

Эндпоинты:
- POST /thick/upload - загрузка .raw файла и анализ
- GET /thick/heatmap/{pipe_id} - получение тепловой карты
- POST /thick/analyze/{pipe_id} - анализ трубы с классификацией
- GET /health - проверка здоровья сервиса
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import numpy as np
import struct
import io
import os
import tempfile
from pathlib import Path
import base64
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ThickService Microservice",
    description="Микросервис для анализа данных толщинометрии труб",
    version="1.0.0"
)

# Пути к директориям
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
HEATMAP_DIR = BASE_DIR / "heatmaps"
MODEL_DIR = BASE_DIR / "model_experiments" / "models"

# Создаем директории если их нет
UPLOAD_DIR.mkdir(exist_ok=True)
HEATMAP_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Путь к модели классификации
MODEL_PATH = MODEL_DIR / "OneDCNNXGBoost_N3200_BS32_DO0.3_LR0.0005_RS42_Augtrue.pth"


# ============================================================================
# Pydantic модели для API
# ============================================================================

class AnalysisResult(BaseModel):
    """Результат анализа трубы"""
    pipe_id: Optional[int] = None
    status: str  # "APPROVED" или "REJECTED"
    confidence: float  # Уверенность классификации (0-1)
    defect_percentage: float  # Процент дефектных областей
    average_thickness: float  # Средняя толщина
    min_thickness: float  # Минимальная толщина
    max_thickness: float  # Максимальная толщина
    thicknom: float  # Номинальная толщина
    matrix_shape: List[int]  # Размеры матрицы [x, y]
    prediction_class: int  # Класс предсказания (0 - норма, 1 - дефект)
    message: Optional[str] = None


class HealthStatus(BaseModel):
    """Статус здоровья сервиса"""
    status: str
    timestamp: str
    model_loaded: bool
    uploads_dir: str
    heatmaps_dir: str


# ============================================================================
# Функции для чтения .raw файлов (из readAndWatchData.ipynb)
# ============================================================================

def read_raw_file(filepath: str) -> Dict[str, Any]:
    """
    Чтение бинарного .raw файла с данными толщинометрии.
    Формат файла (Java DataOutputStream, big-endian):
    - 4 байта: x (int) - количество каналов
    - 4 байта: y (int) - количество точек на канал
    - 8 байт: thicknom (double) - номинальная толщина
    - 1 байт: defective (byte) - флаг дефекта (опционально)
    - x*y*8 байт: matrix (double[]) - данные толщины
    
    Возвращает dict с ключами:
    - 'matrix': numpy array [x, y] с данными толщины
    - 'thicknom': номинальная толщина
    - 'defective': флаг дефекта (если есть в файле)
    - 'x', 'y': размеры матрицы
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f'File not found: {path}')
    
    with open(path, 'rb') as f:
        data = f.read()
    
    if len(data) < 16:
        raise ValueError('File too short - invalid header')
    
    # Чтение заголовка (big-endian как в Java)
    x = struct.unpack_from('>i', data, 0)[0]
    y = struct.unpack_from('>i', data, 4)[0]
    thicknom = struct.unpack_from('>d', data, 8)[0]
    
    # Проверка наличия флага defective (формат из TestingAI.ipynb)
    defective = 0
    data_offset = 16
    if len(data) >= 17:
        # Пробуем прочитать байт defective
        potential_defective = struct.unpack_from('>B', data, 16)[0]
        # Если после 17 байт остается ровно x*y*8 байт, значит это defective флаг
        expected_data_size = x * y * 8
        if len(data) == 17 + expected_data_size:
            defective = potential_defective
            data_offset = 17
        elif len(data) == 16 + expected_data_size:
            # Нет флага defective, определяем по имени файла или thicknom
            defective = 1 if 'defect' in path.name.lower() or thicknom < 0 else 0
    
    # Чтение матрицы данных
    num_values = x * y
    expected_size = data_offset + num_values * 8
    
    if len(data) < expected_size:
        raise ValueError(f'File too short for {x}x{y} matrix. Expected {expected_size} bytes, got {len(data)}')
    
    # Чтение double значений (big-endian)
    matrix_flat = struct.unpack_from(f'>{num_values}d', data, data_offset)
    matrix_np = np.array(matrix_flat, dtype=np.float32).reshape(x, y)
    
    return {
        'matrix': matrix_np,
        'thicknom': thicknom,
        'defective': defective,
        'x': x,
        'y': y
    }


def read_raw_file_from_bytes(file_bytes: bytes) -> Dict[str, Any]:
    """
    Чтение .raw файла из байтов (для загруженных через API файлов)
    """
    if len(file_bytes) < 16:
        raise ValueError('File too short - invalid header')
    
    # Чтение заголовка
    x = struct.unpack_from('>i', file_bytes, 0)[0]
    y = struct.unpack_from('>i', file_bytes, 4)[0]
    thicknom = struct.unpack_from('>d', file_bytes, 8)[0]
    
    # Определение формата
    defective = 0
    data_offset = 16
    num_values = x * y
    expected_size_with_flag = 17 + num_values * 8
    expected_size_without_flag = 16 + num_values * 8
    
    if len(file_bytes) == expected_size_with_flag:
        defective = struct.unpack_from('>B', file_bytes, 16)[0]
        data_offset = 17
    elif len(file_bytes) >= expected_size_without_flag:
        defective = 0
    
    # Чтение матрицы
    matrix_flat = struct.unpack_from(f'>{num_values}d', file_bytes, data_offset)
    matrix_np = np.array(matrix_flat, dtype=np.float32).reshape(x, y)
    
    return {
        'matrix': matrix_np,
        'thicknom': thicknom,
        'defective': defective,
        'x': x,
        'y': y
    }


# ============================================================================
# Функции для создания тепловой карты (из readAndWatchData.ipynb)
# ============================================================================

import matplotlib
matplotlib.use('Agg')  # Не-GUI бэкенд для сервера
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def create_defect_cmap(threshold: float = 1.0) -> mcolors.ListedColormap:
    """
    Создание цветовой карты для визуализации дефектов:
    - 0 → белый
    - значения < порога → градиент красного (от белого к красному)
    - значения > порога → градиент зелёного (от светло-зелёного к насыщенному)
    """
    colors = []
    
    # Белая точка для нуля
    colors.append((1.0, 1.0, 1.0))  # белый
    
    # Градиент красного для значений от 0 до threshold
    n_red = int(256 * threshold) if threshold < 1 else 128
    n_green = int(256 * (1 - threshold)) if threshold < 1 else 128
    
    for i in range(max(1, n_red)):
        intensity = 0.95 - 0.95 * (i / max(1, n_red - 1))
        colors.append((1.0, intensity, intensity))  # розовый/красный градиент
    
    # Насыщенный красный на пороге
    colors.append((1.0, 0.0, 0.0))  # чистый красный
    
    # Градиент зелёного для значений > threshold
    for i in range(max(1, n_green)):
        intensity = 0.5 + 0.5 * (i / max(1, n_green - 1))
        colors.append((0, intensity, 0))  # зелёный градиент
    
    return mcolors.ListedColormap(colors)


def generate_heatmap_image(
    data_2d: np.ndarray,
    thicknom: float,
    output_path: Optional[str] = None,
    title: str = "Толщинометрия"
) -> bytes:
    """
    Генерация тепловой карты толщинометрии.
    
    Args:
        data_2d: 2D массив [X][Y] с данными толщины
        thicknom: номинальная толщина трубы
        output_path: путь для сохранения (опционально)
        title: заголовок графика
    
    Returns:
        bytes: PNG изображение в формате байтов
    """
    x, y = data_2d.shape
    
    # Порог дефекта = 90% от номинальной толщины
    threshold = thicknom * 0.9
    
    # Нормализация для цветовой карты
    vmin = min(0, np.min(data_2d))
    vmax = max(thicknom * 1.1, np.max(data_2d))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # Создание цветовой карты
    cmap = create_defect_cmap(threshold / vmax if vmax > 0 else 0.5)
    
    # Построение изображения
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Транспонирование для корректной ориентации
    im = ax.imshow(
        data_2d.T, 
        cmap=cmap, 
        norm=norm,
        origin='lower', 
        interpolation='bilinear',
        extent=[0, x, 0, y * 5]
    )
    
    # Цветовая шкала
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Маркер порога на colorbar
    cbar.ax.axhline(y=threshold, color='black', linestyle='--', linewidth=1)
    cbar.ax.text(
        2.02, threshold,
        f'Порог: {threshold:.2f}',
        va='center', fontsize=9
    )
    
    # Заголовок
    plt.title(f'{title}\n(цвет: 🟢 норма > {threshold:.2f} > 🔴 дефект)')
    plt.xlabel(f'X (каналы: 0–{x})')
    plt.ylabel(f'Y (точки: 0–{y*5})')
    
    # Сетка
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Статистика на графике
    defect_count = np.sum(data_2d < threshold)
    total_cells = x * y
    defect_pct = 100 * defect_count / total_cells if total_cells > 0 else 0
    
    stats_text = (
        f"Среднее: {np.mean(data_2d):.3f}\n"
        f"Мин: {np.min(data_2d):.3f}\n"
        f"Макс: {np.max(data_2d):.3f}\n"
        f"⚠ <порога: {defect_count}/{total_cells} "
        f"({defect_pct:.1f}%)"
    )
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props
    )
    
    plt.tight_layout()
    
    # Сохранение в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    image_bytes = buf.getvalue()
    
    # Сохранение на диск если указан путь
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(image_bytes)
        logger.info(f"Heatmap saved to: {output_path}")
    
    plt.close(fig)
    return image_bytes


# ============================================================================
# Модель классификации (архитектура из TestingAI.ipynb)
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis


class PipeCNNEncoder(nn.Module):
    """CNN энкодер для извлечения признаков (из TestingAI.ipynb)"""
    
    def __init__(self, input_channels: int = 8, embedding_dim: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.block = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.attention = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        # x: (Batch, Length, Channels) -> (Batch, Channels, Length)
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.block(x)
        w = self.attention(x)
        w = torch.softmax(w, dim=2)
        x = (x * w).sum(dim=2)
        return x


class OneDCNNXGBoostClassifier:
    """
    Гибридная модель классификации: 1D CNN + ручные признаки.
    Упрощенная версия модели из TestingAI.ipynb для инференса.
    """
    
    def __init__(self, input_channels: int = 8, embedding_dim: int = 64, 
                 dropout: float = 0.3, device: str = 'cpu'):
        self.device = device
        self.embedding_dim = embedding_dim
        self.input_channels = input_channels
        
        # CNN энкодер
        self.encoder = PipeCNNEncoder(
            input_channels=input_channels, 
            embedding_dim=embedding_dim
        ).to(device)
        
        # Классификатор на основе эмбеддингов
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim + 6, 32),  # +6 для ручных признаков
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)  # 2 класса: норма/дефект
        ).to(device)
        
        self.scaler = StandardScaler()
        self.is_loaded = False
    
    def load_model(self, model_path: str) -> bool:
        """Загрузка весов модели"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
                elif 'encoder_state_dict' in checkpoint:
                    self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
                else:
                    # Пробуем загрузить напрямую
                    state_dict = {k: v for k, v in checkpoint.items() 
                                  if not k.startswith('xgb')}
                    self.encoder.load_state_dict(state_dict, strict=False)
            else:
                self.encoder.load_state_dict(checkpoint, strict=False)
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load model from {model_path}: {e}")
            self.is_loaded = False
            return False
    
    def extract_manual_features(self, matrix: np.ndarray, thicknom: float) -> np.ndarray:
        """
        Извлечение ручных признаков из матрицы толщины.
        Аналогично build_feature_dataset из TestingAI.ipynb
        """
        features = []
        
        # Статистики матрицы
        features.append(np.mean(matrix))
        features.append(np.std(matrix))
        features.append(np.min(matrix))
        features.append(np.max(matrix))
        features.append(skew(matrix.flatten()))
        features.append(kurtosis(matrix.flatten()))
        
        return np.array(features, dtype=np.float32)
    
    def predict(self, matrix: np.ndarray, thicknom: float) -> Dict[str, Any]:
        """
        Предсказание класса для одной трубы.
        
        Returns:
            dict с ключами:
            - 'class': предсказанный класс (0 или 1)
            - 'confidence': уверенность (0-1)
            - 'probabilities': вероятности классов
        """
        self.encoder.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            # Подготовка данных для CNN
            # matrix: (X, Y) -> нужно (1, Y, X) для формата (Batch, Length, Channels)
            if matrix.shape[0] > matrix.shape[1]:
                # X > Y, значит X это длина, Y это каналы
                data_tensor = torch.FloatTensor(matrix.T).unsqueeze(0).to(self.device)
            else:
                # Y >= X, значит Y это длина, X это каналы
                data_tensor = torch.FloatTensor(matrix).T.unsqueeze(0).to(self.device)
            
            # Извлечение CNN эмбеддингов
            embeddings = self.encoder(data_tensor)
            
            # Извлечение ручных признаков
            manual_feats = self.extract_manual_features(matrix, thicknom)
            manual_feats = torch.FloatTensor(manual_feats).unsqueeze(0).to(self.device)
            
            # Конкатенация признаков
            combined = torch.cat([embeddings, manual_feats], dim=1)
            
            # Классификация
            logits = self.classifier(combined)
            probabilities = torch.softmax(logits, dim=1)
            
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, pred_class].item()
            
            return {
                'class': pred_class,
                'confidence': confidence,
                'probabilities': probabilities[0].cpu().numpy().tolist()
            }
    
    def predict_simple(self, matrix: np.ndarray, thicknom: float) -> Dict[str, Any]:
        """
        Упрощенное предсказание без загрузки модели.
        Использует эвристический анализ данных.
        """
        # Расчет статистик
        mean_val = np.mean(matrix)
        std_val = np.std(matrix)
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        
        # Порог дефекта
        threshold = thicknom * 0.9
        
        # Процент областей ниже порога (потенциальные дефекты)
        defect_cells = np.sum(matrix < threshold)
        total_cells = matrix.size
        defect_pct = defect_cells / total_cells if total_cells > 0 else 0
        
        # Эвристическая классификация
        # Если больше 30% области ниже порога - дефект
        if defect_pct > 0.3:
            pred_class = 1
            confidence = min(0.95, 0.5 + defect_pct)
        else:
            pred_class = 0
            confidence = min(0.95, 0.5 + (1 - defect_pct))
        
        return {
            'class': pred_class,
            'confidence': confidence,
            'probabilities': [1 - confidence, confidence] if pred_class == 1 else [confidence, 1 - confidence],
            'defect_percentage': defect_pct * 100
        }


# Глобальный экземпляр модели
classifier: Optional[OneDCNNXGBoostClassifier] = None


def init_classifier():
    """Инициализация классификатора"""
    global classifier
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    classifier = OneDCNNXGBoostClassifier(
        input_channels=8,
        embedding_dim=64,
        dropout=0.3,
        device=device
    )
    
    # Попытка загрузить модель
    if MODEL_PATH.exists():
        classifier.load_model(str(MODEL_PATH))
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}. Using heuristic predictions.")
    
    return classifier


# ============================================================================
# API эндпоинты
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    logger.info("Starting ThickService Microservice...")
    init_classifier()
    logger.info("ThickService Microservice started successfully")


@app.get("/health", response_model=HealthStatus)
def health_check():
    """Проверка здоровья сервиса"""
    return HealthStatus(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=classifier.is_loaded if classifier else False,
        uploads_dir=str(UPLOAD_DIR),
        heatmaps_dir=str(HEATMAP_DIR)
    )


@app.post("/thick/upload", response_model=AnalysisResult)
async def upload_raw_file(file: UploadFile = File(...), pipe_id: Optional[int] = None):
    """
    Загрузка .raw файла и выполнение полного анализа.
    
    Эквивалент комбинации endpoints из ThickController:
    - exportThickToRaw (загрузка файла)
    - analyzeData (классификация)
    - generateHeatmap (генерация тепловой карты)
    """
    try:
        # Чтение файла
        contents = await file.read()
        
        if len(contents) < 16:
            raise HTTPException(status_code=400, detail="Invalid file: too small")
        
        # Парсинг .raw файла
        raw_data = read_raw_file_from_bytes(contents)
        matrix = raw_data['matrix']
        thicknom = raw_data['thicknom']
        
        logger.info(f"Uploaded file: {file.filename}, shape: {matrix.shape}, thicknom: {thicknom}")
        
        # Генерация тепловой карты
        heatmap_bytes = generate_heatmap_image(
            matrix, 
            thicknom,
            title=f"Pipe ID: {pipe_id or 'Unknown'}"
        )
        
        # Сохранение heatmap
        heatmap_filename = f"heatmap_{pipe_id or datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        heatmap_path = HEATMAP_DIR / heatmap_filename
        with open(heatmap_path, 'wb') as f:
            f.write(heatmap_bytes)
        
        # Классификация
        if classifier and classifier.is_loaded:
            prediction = classifier.predict(matrix, thicknom)
        else:
            prediction = classifier.predict_simple(matrix, thicknom) if classifier else {'class': 0, 'confidence': 0.5, 'defect_percentage': 0}
        
        # Расчет метрик
        threshold = thicknom * 0.9
        defect_cells = np.sum(matrix < threshold)
        defect_pct = float(defect_cells / matrix.size * 100) if matrix.size > 0 else 0
        
        # Определение статуса
        pred_class = prediction.get('class', 0)
        status = "REJECTED" if pred_class == 1 else "APPROVED"
        
        result = AnalysisResult(
            pipe_id=pipe_id,
            status=status,
            confidence=prediction.get('confidence', 0.5),
            defect_percentage=defect_pct,
            average_thickness=float(np.mean(matrix)),
            min_thickness=float(np.min(matrix)),
            max_thickness=float(np.max(matrix)),
            thicknom=thicknom,
            matrix_shape=[matrix.shape[0], matrix.shape[1]],
            prediction_class=pred_class,
            message=f"Heatmap saved to: {heatmap_path}"
        )
        
        logger.info(f"Analysis complete: {status}, confidence: {result.confidence:.2f}")
        
        return result
        
    except ValueError as e:
        logger.error(f"Invalid file format: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid file format: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/thick/heatmap/{pipe_id}")
async def get_heatmap(pipe_id: int):
    """
    Получение тепловой карты для указанной трубы.
    Эквивалент GET /thick/heatmap/{pipeId} из ThickController
    """
    # Поиск файла heatmap
    heatmap_path = HEATMAP_DIR / f"heatmap_{pipe_id}.png"
    
    if not heatmap_path.exists():
        # Пробуем другие форматы именования
        possible_files = list(HEATMAP_DIR.glob(f"*{pipe_id}*.png"))
        if possible_files:
            heatmap_path = possible_files[0]
        else:
            raise HTTPException(status_code=404, detail=f"Heatmap not found for pipe ID: {pipe_id}")
    
    with open(heatmap_path, 'rb') as f:
        image_bytes = f.read()
    
    return Response(
        content=image_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": f"attachment; filename=\"heatmap_{pipe_id}.png\""
        }
    )


@app.post("/thick/analyze/{pipe_id}", response_model=AnalysisResult)
async def analyze_pipe(pipe_id: int, file: Optional[UploadFile] = File(None)):
    """
    Анализ трубы с классификацией.
    Эквивалент POST /thick/analyze/{pipeId} из ThickController
    
    Если файл предоставлен - анализирует его.
    Если нет - ищет существующие данные для pipe_id.
    """
    try:
        matrix = None
        thicknom = None
        
        if file:
            # Анализ загруженного файла
            contents = await file.read()
            raw_data = read_raw_file_from_bytes(contents)
            matrix = raw_data['matrix']
            thicknom = raw_data['thicknom']
        else:
            # Поиск существующего файла
            raw_path = UPLOAD_DIR / f"thick_{pipe_id}.raw"
            if not raw_path.exists():
                # Пробуем найти в других директориях
                for search_dir in [BASE_DIR, BASE_DIR / "thicks_export", BASE_DIR / "dataset"]:
                    possible_files = list(search_dir.glob(f"thick_{pipe_id}.raw"))
                    if possible_files:
                        raw_path = possible_files[0]
                        break
            
            if not raw_path.exists():
                raise HTTPException(status_code=404, detail=f"Raw file not found for pipe ID: {pipe_id}")
            
            raw_data = read_raw_file(str(raw_path))
            matrix = raw_data['matrix']
            thicknom = raw_data['thicknom']
        
        # Классификация
        if classifier and classifier.is_loaded:
            prediction = classifier.predict(matrix, thicknom)
        else:
            prediction = classifier.predict_simple(matrix, thicknom) if classifier else {'class': 0, 'confidence': 0.5, 'defect_percentage': 0}
        
        # Расчет метрик
        threshold = thicknom * 0.9
        defect_cells = np.sum(matrix < threshold)
        defect_pct = float(defect_cells / matrix.size * 100) if matrix.size > 0 else 0
        
        pred_class = prediction.get('class', 0)
        status = "REJECTED" if pred_class == 1 else "APPROVED"
        
        # Генерация heatmap если еще нет
        heatmap_path = HEATMAP_DIR / f"heatmap_{pipe_id}.png"
        if not heatmap_path.exists():
            heatmap_bytes = generate_heatmap_image(matrix, thicknom, title=f"Pipe ID: {pipe_id}")
            with open(heatmap_path, 'wb') as f:
                f.write(heatmap_bytes)
        
        result = AnalysisResult(
            pipe_id=pipe_id,
            status=status,
            confidence=prediction.get('confidence', 0.5),
            defect_percentage=defect_pct,
            average_thickness=float(np.mean(matrix)),
            min_thickness=float(np.min(matrix)),
            max_thickness=float(np.max(matrix)),
            thicknom=thicknom,
            matrix_shape=[matrix.shape[0], matrix.shape[1]],
            prediction_class=pred_class,
            message=f"Heatmap available at: /thick/heatmap/{pipe_id}"
        )
        
        logger.info(f"Analysis complete for pipe {pipe_id}: {status}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing pipe {pipe_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analyzing pipe: {str(e)}")


@app.get("/thick/raw/{pipe_id}")
async def get_raw_file(pipe_id: int):
    """
    Получение raw файла для указанной трубы.
    Эквивалент GET /thick/raw/{pipeId} из ThickController
    """
    # Поиск файла
    raw_path = UPLOAD_DIR / f"thick_{pipe_id}.raw"
    
    if not raw_path.exists():
        # Пробуем найти в других директориях
        for search_dir in [BASE_DIR, BASE_DIR / "thicks_export", BASE_DIR / "dataset"]:
            possible_files = list(search_dir.glob(f"thick_{pipe_id}.raw"))
            if possible_files:
                raw_path = possible_files[0]
                break
    
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail=f"Raw file not found for pipe ID: {pipe_id}")
    
    with open(raw_path, 'rb') as f:
        file_bytes = f.read()
    
    return Response(
        content=file_bytes,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename=\"thick_{pipe_id}.raw\""
        }
    )


@app.post("/thick/analyze-with-image", response_model=Dict[str, Any])
async def analyze_with_image(file: UploadFile = File(...), pipe_id: Optional[int] = None):
    """
    Комбинированный эндпоинт: загрузка файла + анализ + возврат изображения.
    Возвращает результат анализа и тепловую карту в одном запросе.
    """
    try:
        contents = await file.read()
        raw_data = read_raw_file_from_bytes(contents)
        matrix = raw_data['matrix']
        thicknom = raw_data['thicknom']
        
        # Генерация heatmap
        heatmap_bytes = generate_heatmap_image(
            matrix, thicknom,
            title=f"Pipe ID: {pipe_id or 'Unknown'}"
        )
        
        # Кодирование изображения в base64
        image_base64 = base64.b64encode(heatmap_bytes).decode('utf-8')
        
        # Классификация
        if classifier and classifier.is_loaded:
            prediction = classifier.predict(matrix, thicknom)
        else:
            prediction = classifier.predict_simple(matrix, thicknom) if classifier else {'class': 0, 'confidence': 0.5}
        
        threshold = thicknom * 0.9
        defect_pct = float(np.sum(matrix < threshold) / matrix.size * 100) if matrix.size > 0 else 0
        
        pred_class = prediction.get('class', 0)
        status = "REJECTED" if pred_class == 1 else "APPROVED"
        
        return {
            "pipe_id": pipe_id,
            "status": status,
            "confidence": prediction.get('confidence', 0.5),
            "defect_percentage": defect_pct,
            "average_thickness": float(np.mean(matrix)),
            "min_thickness": float(np.min(matrix)),
            "max_thickness": float(np.max(matrix)),
            "thicknom": thicknom,
            "prediction_class": pred_class,
            "heatmap_base64": f"data:image/png;base64,{image_base64}"
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_with_image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)