"""Configuration schema and loader for the projector mapping pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, validator


class CameraConfig(BaseModel):
    device_id: int = Field(..., ge=0)
    resolution: List[int] = Field(..., min_items=2, max_items=2)
    fps: int = Field(30, gt=0)

    @property
    def width(self) -> int:
        return self.resolution[0]

    @property
    def height(self) -> int:
        return self.resolution[1]


class ProjectorConfig(BaseModel):
    id: str
    display_id: int = Field(..., ge=0)
    resolution: List[int] = Field(..., min_items=2, max_items=2)
    overlay_color_lower_hsv: List[int] = Field(default_factory=lambda: [0, 120, 70], min_items=3, max_items=3)
    overlay_color_upper_hsv: List[int] = Field(default_factory=lambda: [10, 255, 255], min_items=3, max_items=3)
    overlay_color_bgr: List[int] = Field(default_factory=lambda: [0, 0, 255], min_items=3, max_items=3)

    @property
    def width(self) -> int:
        return self.resolution[0]

    @property
    def height(self) -> int:
        return self.resolution[1]


class DetectionAPIConfig(BaseModel):
    endpoint: str
    api_key_env: str


class DetectionConfig(BaseModel):
    backend: str
    classes: List[str]
    confidence_threshold: float = Field(0.4, ge=0.0, le=1.0)
    api: DetectionAPIConfig
    evfsam: Optional["EvfSamConfig"] = None

    @validator("evfsam", always=True)
    def ensure_prompt_map(cls, value: Optional["EvfSamConfig"], values: Dict[str, object]) -> Optional["EvfSamConfig"]:
        if value is None:
            return value
        classes: List[str] = values.get("classes", [])  # type: ignore[assignment]
        missing = [label for label in classes if label not in value.prompt_map]
        if missing:
            raise ValueError(f"EVF-SAM prompt_map missing labels: {missing}")
        return value


class MaskingConfig(BaseModel):
    dilate_kernel: int = Field(5, ge=1)
    blur_kernel: int = Field(7, ge=1)


class AlignmentConfig(BaseModel):
    max_iterations: int = Field(5, ge=1)
    reprojection_tolerance_px: float = Field(5.0, gt=0)
    feedback_gain: float = Field(0.6, ge=0.0, le=1.0)
    min_overlay_contour_area: int = Field(300, ge=1)
    max_polygon_points: int = Field(8, ge=4)


class EvfSamConfig(BaseModel):
    model_id: str = Field("fal-ai/evf-sam")
    api_key_env: str = Field("FALAI_API_KEY")
    prompt_map: Dict[str, str] = Field(default_factory=dict)
    mask_only: bool = True
    fill_holes: bool = True
    revert_mask: bool = False
    poll_interval_s: float = Field(0.5, gt=0.0)
    timeout_s: float = Field(20.0, gt=0.0)


class CalibrationPatternConfig(BaseModel):
    type: str = Field("checkerboard")
    inner_corners: List[int] = Field(..., min_items=2, max_items=2)
    square_size_m: float = Field(..., gt=0.0)


class CalibrationPaths(BaseModel):
    camera_intrinsics_path: Path
    projector_intrinsics_dir: Path
    homographies_dir: Path

    @validator("projector_intrinsics_dir", "homographies_dir", pre=True)
    def ensure_dir(cls, value: str | Path) -> Path:
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        return path


class CalibrationConfig(BaseModel):
    pattern: CalibrationPatternConfig
    camera_intrinsics_path: Path
    projector_intrinsics_dir: Path
    homographies_dir: Path

    @validator("camera_intrinsics_path", pre=True)
    def ensure_parent(cls, value: str | Path) -> Path:
        path = Path(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @validator("projector_intrinsics_dir", "homographies_dir", pre=True)
    def ensure_dirs(cls, value: str | Path) -> Path:
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        return path


class LoggingConfig(BaseModel):
    level: str = Field("INFO")
    output: str = Field("stdout")


class StorageConfig(BaseModel):
    frames_dir: Path
    masks_dir: Path
    logs_dir: Path

    @validator("frames_dir", "masks_dir", "logs_dir", pre=True)
    def ensure_storage_dirs(cls, value: str | Path) -> Path:
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        return path


class HardwareConfig(BaseModel):
    camera: CameraConfig
    projectors: List[ProjectorConfig]


class ProjectMetadata(BaseModel):
    name: str
    data_root: Path

    @validator("data_root", pre=True)
    def ensure_data_root(cls, value: str | Path) -> Path:
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        return path


class PipelineModuleConfig(BaseModel):
    detection: DetectionConfig
    masking: MaskingConfig
    alignment: AlignmentConfig


class PipelineConfig(BaseModel):
    project: ProjectMetadata
    logging: LoggingConfig
    hardware: HardwareConfig
    pipeline: PipelineModuleConfig
    calibration: CalibrationConfig
    storage: StorageConfig

    def projector_by_id(self, projector_id: str) -> ProjectorConfig:
        for projector in self.hardware.projectors:
            if projector.id == projector_id:
                return projector
        raise KeyError(f"Projector '{projector_id}' not found in configuration")


def load_config(path: str | Path) -> PipelineConfig:
    """Load configuration from YAML file."""
    with open(path, "r", encoding="utf-8") as handle:
        raw: Dict[str, object] = yaml.safe_load(handle)
    return PipelineConfig.model_validate(raw)
