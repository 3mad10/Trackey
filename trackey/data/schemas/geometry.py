from pydantic import BaseModel
from typing import List, Tuple


class Point(BaseModel):
    """Point in normalized coordinates (0-1)"""
    x: float
    y: float
    
    def __iter__(self):
        """Allow unpacking: x, y = point"""
        return iter((self.x, self.y))


class Polygon(BaseModel):
    """Polygon defined by vertices"""
    points: List[Tuple[float, float]]  # List of (x, y) normalized
    
    def contains(self, point: Tuple[float, float]) -> bool:
        """Check if point is inside polygon using ray casting"""
        x, y = point
        n = len(self.points)
        inside = False
        
        p1x, p1y = self.points[0]
        for i in range(1, n + 1):
            p2x, p2y = self.points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside


class Line(BaseModel):
    """Line segment"""
    name: str
    start: Tuple[float, float]  # (x, y) normalized
    end: Tuple[float, float]    # (x, y) normalized
    
    def crossed(self, prev_point: Tuple[float, float], 
                curr_point: Tuple[float, float]) -> bool:
        """Check if movement from prev to curr crossed this line"""
        # Line intersection algorithm
        x1, y1 = prev_point
        x2, y2 = curr_point
        x3, y3 = self.start
        x4, y4 = self.end
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return False
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        return 0 <= t <= 1 and 0 <= u <= 1


class Zone(BaseModel):
    """Named region for analysis"""
    name: str
    polygon: Polygon
    color: Tuple[int,int,int] = (128,128,128)
    alpha: float = 0.2
    filled: bool = True
    
    def contains(self, point: Tuple[float, float]) -> bool:
        """Check if point is in zone"""
        return self.polygon.contains(point)