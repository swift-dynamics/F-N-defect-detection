from dataclasses import dataclass

@dataclass
class ROICoordinates:
    x: int
    y: int
    width: int
    height: int